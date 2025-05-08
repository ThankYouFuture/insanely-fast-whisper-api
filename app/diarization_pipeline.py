import torch
from pyannote.audio import Pipeline

from .diarize import (
    post_process_segments_and_transcripts,
    diarize_audio,
    preprocess_inputs,
)


# Déterminer le périphérique à utiliser pour la diarisation
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    diarization_device = torch.device("mps")
    print("Diarization using MPS device")
elif torch.cuda.is_available():
    diarization_device = torch.device("cuda:0")
    print("Diarization using CUDA device")
else:
    diarization_device = torch.device("cpu")
    print("Diarization using CPU device")


def diarize(hf_token, file_name, outputs):
    diarization_pipeline = Pipeline.from_pretrained(
        checkpoint_path="pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    diarization_pipeline.to(diarization_device)

    inputs, diarizer_inputs = preprocess_inputs(inputs=file_name)

    segments = diarize_audio(diarizer_inputs, diarization_pipeline)
    return post_process_segments_and_transcripts(
        segments, outputs["chunks"], group_by_speaker=False
    )
