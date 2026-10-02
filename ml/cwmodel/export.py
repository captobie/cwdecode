"""Export a checkpoint for the app: the Core ML model plus golden test data.

Runs in the export environment (Python 3.13 with the torch version coremltools is tested
against), because coremltools' native parts aren't built for newer Pythons:

    /opt/homebrew/bin/python3.13 -m venv .venv-export
    .venv-export/bin/pip install "torch==2.7.0" coremltools numpy scipy && .venv-export/bin/pip install -e . --no-deps
    .venv-export/bin/python -m cwmodel.export --ckpt runs/full/best.pt

Writes Sources/CWKit/Resources/CWNet.mlmodelc (compiled with Xcode's coremlcompiler, since
SwiftPM can't compile an .mlpackage resource) and Tests/CWKitTests/Resources/Neural/. The golden
files are synthetic (never W1AW audio) and pin the Swift front end, model and streaming decoder to this
Python reference.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from cwmodel import features
from cwmodel.evaluate import load_model
from cwmodel.net import TIME_STRIDE, count_parameters
from cwmodel.stream import OUTPUT_FRAME_S, StreamingDecoder, decode_session, torch_log_probs
from cwsynth.alphabet import TOKENS
from cwsynth.render import render
from cwsynth.spec import QsbParams, SampleSpec, sample_fist

REPO = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO / "Sources" / "CWKit" / "Resources" / "CWNet.mlmodelc"
GOLDEN_DIR = REPO / "Tests" / "CWKitTests" / "Resources" / "Neural"
STREAM = {"window_s": 6.0, "hop_s": 1.0, "context_s": 2.5, "chunk_s": 0.1}
PCM_SCALE = 32768.0


def feature_metadata() -> dict:
    return {"sample_rate": features.SAMPLE_RATE, "n_fft": features.N_FFT, "hop": features.HOP,
            "bin_lo": features.BIN_LO, "bin_hi": features.BIN_HI,
            "dynamic_range_db": features.DYNAMIC_RANGE_DB, "time_stride": TIME_STRIDE,
            "output_frame_s": OUTPUT_FRAME_S}


def convert(model: torch.nn.Module, checkpoint: dict, source: str):
    import coremltools as ct

    class BatchFirst(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, spec):
            return self.net(spec).permute(1, 0, 2)   # [1, frames, classes]

    traced = torch.jit.trace(BatchFirst(model).eval(), torch.zeros(1, 1, features.N_BINS, 1000))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="spectrogram",
                              shape=(1, 1, features.N_BINS, ct.RangeDim(64, 8192, default=1000)))],
        outputs=[ct.TensorType(name="log_probs")],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
        convert_to="mlprogram",
    )
    metrics = checkpoint.get("metrics", {})
    mlmodel.short_description = "CW (Morse) decoder: spectrogram → per-frame token log-probabilities (CTC)"
    mlmodel.version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    mlmodel.user_defined_metadata.update({
        "vocabulary": json.dumps(TOKENS),
        "features": json.dumps(feature_metadata()),
        "source_checkpoint": source,
        "training_step": str(checkpoint.get("step")),
        "parameters": str(count_parameters(model)),
        "w1aw_cer": f"{metrics['w1aw']['cer']:.4f}" if "w1aw" in metrics else "",
        "synthetic_val_cer": f"{metrics['val']['cer']:.4f}" if "val" in metrics else "",
    })
    return mlmodel


def save_compiled(mlmodel) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / "CWNet.mlpackage"
        mlmodel.save(str(package))
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), tmp],
                       check=True, stdout=subprocess.DEVNULL)
        shutil.rmtree(MODEL_PATH, ignore_errors=True)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(tmp) / "CWNet.mlmodelc", MODEL_PATH)


def quantize(audio: np.ndarray) -> np.ndarray:
    return np.clip(np.round(audio * PCM_SCALE), -32768, 32767).astype("<i2")


def golden_clips() -> dict[str, SampleSpec]:
    rng = np.random.default_rng(2026)
    return {
        "features": SampleSpec(text="CQ DE K1ABC", seed=11, wpm=24, tone_hz=640, snr_db=8.0,
                               lead_s=0.3, tail_s=0.3, gain_peak=0.6),
        "stream_hand": SampleSpec(
            text="K1ABC DE W1XYZ <BT> TNX FER CALL <BT> UR RST 579 579 <BT> QTH BOSTON <BT> "
                 "NAME CARL CARL <BT> HW? K1ABC DE W1XYZ <KN>",
            seed=12, wpm=20, tone_hz=700, snr_db=10.0, fist=sample_fist(rng, "straight"),
            lead_s=1.0, tail_s=1.5, gain_peak=0.5),
        "stream_fast": SampleSpec(text="CQ TEST DL5ABC DL5ABC TU 5NN 14 TU CQ TEST DL5ABC " * 3,
                                  seed=13, wpm=32, tone_hz=560, snr_db=6.0,
                                  qsb=QsbParams(0.2, 6.0), lead_s=0.5, tail_s=0.5, gain_peak=0.7),
        "stream_noise": SampleSpec(text="", seed=14, snr_db=0.0, lead_s=0.0, tail_s=12.0),
    }


def write_golden(model: torch.nn.Module) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    log_probs = torch_log_probs(model)
    golden = {"pcm_scale": PCM_SCALE, "stream": STREAM, "features_meta": feature_metadata(), "clips": {}}
    for name, spec in golden_clips().items():
        pcm = quantize(render(spec).audio)
        (GOLDEN_DIR / f"{name}.s16").write_bytes(pcm.tobytes())
        audio = pcm.astype(np.float32) / PCM_SCALE        # exactly what Swift will read
        entry = {"samples": len(pcm), "reference": spec.text}
        if name == "features":
            spec_tensor = features.spectrogram(audio)
            lp = log_probs(spec_tensor)
            entry["spectrogram"] = {"bins": spec_tensor.shape[0], "frames": spec_tensor.shape[1],
                                    "values": np.round(spec_tensor.numpy(), 6).ravel().tolist()}
            entry["log_probs_head"] = np.round(lp[:40], 5).tolist()   # first 40 output frames
            entry["output_frames"] = lp.shape[0]
        decoder = StreamingDecoder(log_probs, STREAM["window_s"], STREAM["hop_s"], STREAM["context_s"])
        entry["streamed_text"] = decode_session(decoder, audio, STREAM["chunk_s"])
        golden["clips"][name] = entry
        print(f"  {name:13s} {len(pcm) / features.SAMPLE_RATE:5.1f} s  "
              f"reference {spec.text[:40]!r}  streamed {entry['streamed_text'][:40]!r}")
    (GOLDEN_DIR / "golden.json").write_text(json.dumps(golden, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--golden-only", action="store_true", help="skip the Core ML conversion")
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = load_model(args.ckpt, "cpu").eval()
    if not args.golden_only:
        mlmodel = convert(model, checkpoint, str(args.ckpt))
        save_compiled(mlmodel)
        print(f"wrote {MODEL_PATH.relative_to(REPO)}")
    write_golden(model)
    print(f"wrote {GOLDEN_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
