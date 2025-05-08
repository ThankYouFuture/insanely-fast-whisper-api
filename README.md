# Insanely Fast Whisper API
An API to transcribe audio with [OpenAI's Whisper Large v3](https://huggingface.co/openai/whisper-large-v3)! Powered by 🤗 Transformers and Optimum, with support for Apple Silicon (MPS) and NVIDIA GPUs (utilizing flash-attn where available).

Features:
* 🎤 Transcribe audio to text at blazing fast speeds
* 📖 Fully open source, deployable on GPU cloud providers (for NVIDIA GPUs), and runnable locally on systems with NVIDIA GPUs or Macs with Apple Silicon (MPS).
* 🗣️ Built-in speaker diarization
* ⚡ Easy to use and Fast API layer
* 📃 Async background tasks and webhooks
* 🔥 Optimized for concurrency and parallel processing
* ✅ Task management, cancel and status endpoints
* 🔒 Admin authentication for secure API access
* 🧩 Fully managed API available on [JigsawStack](https://jigsawstack.com/speech-to-text)

Based on [Insanely Fast Whisper CLI](https://github.com/Vaibhavs10/insanely-fast-whisper) project. Check it out if you like to set up this project locally or understand the background of insanely-fast-whisper.

### Apple Silicon (Mac) Support
This project has been adapted to run on Macs with Apple Silicon (M1/M2/M3 series chips) leveraging Metal Performance Shaders (MPS) for hardware acceleration.
When running on MPS:
* The `flash-attn` optimization (specific to NVIDIA GPUs) is not used.
* Performance will vary based on your Mac's specific chip and configuration.
The application dynamically detects available hardware (MPS, CUDA, or CPU) and configures itself accordingly.

This project is focused on providing a deployable blazing fast whisper API with docker. It can be deployed on cloud infrastructure with NVIDIA GPUs for scalable production use cases, or run locally on compatible hardware including Macs.


**Note on Benchmarks:** The benchmarks below were performed on NVIDIA A100-80GB GPUs utilizing Flash Attention 2. These are indicative of performance on high-end NVIDIA hardware. Performance on other hardware, such as Apple Silicon (MPS), will differ.

Here are some benchmarks we ran on Nvidia A100 - 80GB GPUs👇
| Optimization type    | Time to Transcribe (150 mins of Audio) |
|------------------|------------------|
| **large-v3 (Transformers) (`fp16` + `batching [24]` + `Flash Attention 2`)** | **~2 (*1 min 38 sec*)**            |
| **large-v3 (Transformers) (`fp16` + `batching [24]` + `Flash Attention 2` + `diarization`)** | **~2 (*3 min 16 sec*)**            |

## Docker image
```
yoeven/insanely-fast-whisper-api:latest
```
Docker hub: [yoeven/insanely-fast-whisper-api](https://hub.docker.com/r/yoeven/insanely-fast-whisper-api)

## Deploying to other cloud providers
Since this is a dockerized app, you can deploy it to any cloud provider that supports docker and GPUs with a few config tweaks.

## Fully managed and scalable API 
[JigsawStack](https://jigsawstack.com) provides a bunch of powerful APIs for various use cases while keeping costs low. This project is available as a fully managed API [here](https://jigsawstack.com/speech-to-text) with enhanced cloud scalability for cost efficiency and high uptime. Sign up [here](https://jigsawstack.com) for free!


## API usage

### Authentication
If you had set up the `ADMIN_KEY` environment secret. You'll need to pass `x-admin-api-key` in the header with the value of the key you previously set.


### Endpoints
#### Base URL
Depending on the cloud provider you deploy to, the base URL will be different.

#### **POST** `/`
Transcribe or translate audio into text
##### Body params (JSON)
| Name    | value |
|------------------|------------------|
| url (Required) |  url of audio |
| task | `transcribe`, `translate`  default: `transcribe` |
| language | `None`, `en`, [other languages](https://huggingface.co/openai/whisper-large-v3) default: `None` Auto detects language
| batch_size | Number of parallel batches you want to compute. Reduce if you face OOMs. default: `64` |
| timestamp | `chunk`, `word`  default: `chunk` |
| diarise_audio | Diarise the audio clips by speaker. You will need to set hf_token. default:`false` |
| webhook | Webhook `POST` call on completion or error. default: `None` |
| webhook.url | URL to send the webhook |
| webhook.header | Headers to send with the webhook |
| is_async | Run task in background and sends results to webhook URL. `true`, `false` default: `false` |
| managed_task_id | Custom Task ID used to reference ongoing task. default: `uuid() v4 will be generated for each transcription task` |

#### **GET** `/tasks`
Get all active transcription tasks, both async background tasks and ongoing tasks

#### **GET** `/status/{task_id}`
Get the status of a task, completed tasks will be removed from the list which may throw an error

#### **DELETE** `/cancel/{task_id}`
Cancel async background task. Only transcription jobs created with `is_async` set to `true` can be cancelled.


## Running locally
```bash
# clone the repo
$ git clone https://github.com/jigsawstack/insanely-fast-whisper-api.git

# change the working directory
$ cd insanely-fast-whisper-api

# install torch
$ pip3 install torch torchvision torchaudio

# upgrade wheel and install required packages for FlashAttention
$ pip3 install -U wheel && pip install ninja packaging

# install FlashAttention
$ pip3 install flash-attn --no-build-isolation

# generate updated requirements.txt if you want to use other management tools (Optional)
$ poetry export --output requirements.txt

# get the path of python
$ which python3

# setup virtual environment 
$ poetry env use /full/path/to/python

# install the requirements
$ poetry install

# run the app
$ uvicorn app.app:app --reload
```

## Extra

### Webhooks
When using `is_async: true`, the API will immediately return a task ID and process the transcription in the background.
Once completed (or if an error occurs), the API will send a `POST` request to the `webhook.url` you provided.

**Example Webhook Payload (Success):**
```json
{
  "status": "completed",
  "task_id": "your-task-id",
  "result": {
    "text": "Transcription text...",
    "chunks": [
      { "speaker": "SPEAKER_00", "timestamp": [0.0, 5.2], "text": "Hello world" },
      { "speaker": "SPEAKER_01", "timestamp": [5.5, 8.1], "text": "How are you?" }
    ],
    "language": "en"
  }
}
```

**Example Webhook Payload (Error):**
```json
{
  "status": "error",
  "task_id": "your-task-id",
  "error": "Details about the error..."
}
```

### Admin Key
For security, you can set an `ADMIN_KEY` environment variable. If set, all API requests (except `/docs` and `/openapi.json`) will require an `X-Admin-Key` header matching this value.

### Hugging Face Token for Diarization
To use speaker diarization (`diarise_audio: true`), you need a Hugging Face User Access Token with `read` permissions. Set this token in the `HF_TOKEN` environment variable.
You'll also need to accept the user conditions for:
1. [`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0)
2. [`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1)

## Acknowledgements

1. [Vaibhav Srivastav](https://github.com/Vaibhavs10) for writing a huge chunk of the code and the CLI version of this project.
2. [OpenAI Whisper](https://huggingface.co/openai/whisper-large-v3) 


## JigsawStack
This project is part of [JigsawStack](https://jigsawstack.com) - A suite of powerful and developer friendly APIs for various use cases while keeping costs low. Sign up [here](https://jigsawstack.com) for free!
