import os
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Body,
    BackgroundTasks,
    Request,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from transformers import pipeline
from .diarization_pipeline import diarize
import requests
import asyncio
import uuid


admin_key = os.environ.get(
    "ADMIN_KEY",
)

hf_token = os.environ.get(
    "HF_TOKEN",
)


# Déterminer le périphérique à utiliser
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS device")
    # model_kwargs doit être retiré pour MPS
    model_kwargs = {}
    main_torch_dtype = torch.float16 # MPS supporte float16
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using CUDA device")
    # Garder flash_attention_2 si CUDA est disponible et compatible
    model_kwargs={"attn_implementation": "flash_attention_2"}
    main_torch_dtype = torch.float16 # CUDA supporte float16
else:
    device = torch.device("cpu")
    print("Using CPU device")
    # Pas de model_kwargs spécifiques pour CPU, ou ceux compatibles CPU si besoin
    model_kwargs = {}
    main_torch_dtype = torch.float32 # CPU préfère float32 pour la compatibilité/performance

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=main_torch_dtype,
    device=device,
    model_kwargs=model_kwargs if model_kwargs else None, # Ne passer que si non vide
)

app = FastAPI()
loop = asyncio.get_event_loop()
running_tasks = {}


class WebhookBody(BaseModel):
    url: str
    header: dict[str, str] = {}


def process(
    url: str,
    task: str,
    language: str,
    batch_size: int,
    timestamp: str,
    diarise_audio: bool,
    webhook: WebhookBody | None = None,
    task_id: str | None = None,
):
    errorMessage: str | None = None
    outputs = {}
    try:
        generate_kwargs = {
            "task": task,
            "language": None if language == "None" else language,
        }

        outputs = pipe(
            url,
            chunk_length_s=30,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps="word" if timestamp == "word" else True,
        )

        if diarise_audio is True:
            speakers_transcript = diarize(
                hf_token,
                url,
                outputs,
            )
            outputs["speakers"] = speakers_transcript
    except asyncio.CancelledError:
        errorMessage = "Task Cancelled"
    except Exception as e:
        errorMessage = str(e)

    if task_id is not None:
        del running_tasks[task_id]

    if webhook is not None:
        webhookResp = (
            {"output": outputs, "status": "completed", "task_id": task_id}
            if errorMessage is None
            else {"error": errorMessage, "status": "error", "task_id": task_id}
        )

        requests.post(
            webhook.url,
            headers=webhook.header,
            json=(webhookResp),
        )

    if errorMessage is not None:
        raise Exception(errorMessage)

    return outputs


@app.middleware("http")
async def admin_key_auth_check(request: Request, call_next):
    if admin_key is not None:
        if ("x-admin-api-key" not in request.headers) or (
            request.headers["x-admin-api-key"] != admin_key
        ):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    response = await call_next(request)
    return response


@app.post("/")
def root(
    url: str = Body(),
    task: str = Body(default="transcribe", enum=["transcribe", "translate"]),
    language: str = Body(default="None"),
    batch_size: int = Body(default=64),
    timestamp: str = Body(default="chunk", enum=["chunk", "word"]),
    diarise_audio: bool = Body(
        default=False,
    ),
    webhook: WebhookBody | None = None,
    is_async: bool = Body(default=False),
    managed_task_id: str | None = Body(default=None),
):
    if url.lower().startswith("http") is False:
        raise HTTPException(status_code=400, detail="Invalid URL")

    if diarise_audio is True and hf_token is None:
        raise HTTPException(status_code=500, detail="Missing Hugging Face Token")

    if is_async is True and webhook is None:
        raise HTTPException(
            status_code=400, detail="Webhook is required for async tasks"
        )

    task_id = managed_task_id if managed_task_id is not None else str(uuid.uuid4())

    try:
        resp = {}
        if is_async is True:
            backgroundTask = asyncio.ensure_future(
                loop.run_in_executor(
                    None,
                    process,
                    url,
                    task,
                    language,
                    batch_size,
                    timestamp,
                    diarise_audio,
                    webhook,
                    task_id,
                )
            )
            running_tasks[task_id] = backgroundTask
            resp = {
                "detail": "Task is being processed in the background",
                "status": "processing",
                "task_id": task_id,
            }
        else:
            running_tasks[task_id] = None
            outputs = process(
                url,
                task,
                language,
                batch_size,
                timestamp,
                diarise_audio,
                webhook,
                task_id,
            )
            resp = {
                "output": outputs,
                "status": "completed",
                "task_id": task_id,
            }
        return resp
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def tasks():
    return {"tasks": list(running_tasks.keys())}


@app.get("/status/{task_id}")
def status(task_id: str):
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = running_tasks[task_id]

    if task is None:
        return {"status": "processing"}
    elif task.done() is False:
        return {"status": "processing"}
    else:
        return {"status": "completed", "output": task.result()}


@app.delete("/cancel/{task_id}")
def cancel(task_id: str):
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = running_tasks[task_id]
    if task is None:
        return HTTPException(status_code=400, detail="Not a background task")
    elif task.done() is False:
        task.cancel()
        del running_tasks[task_id]
        return {"status": "cancelled"}
    else:
        return {"status": "completed", "output": task.result()}
