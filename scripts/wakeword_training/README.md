# NixOrb wake-word training

Trains custom openWakeWord models for "hey nixorb", "nixorb", and "hypernix"
(or any phrases you pass), overnight, unattended, on an 8 GB GPU.

## Setup

```bash
cd scripts/wakeword_training
python3.12 -m venv .venv-train
source .venv-train/bin/activate
pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Separate venv from NixOrb's own install on purpose — this is a one-off
training environment, not something NixOrb needs at runtime.

## Run it

```bash
python train_wakeword.py                 # 8-hour budget, default phrases
# or, e.g.:
python train_wakeword.py --hours 6
```

Kick it off before bed:

```bash
nohup python train_wakeword.py > /dev/null 2>&1 &
disown
```

Progress goes to `~/.local/share/nixorb/wakeword_training/train.log` (and
stdout). Safe to Ctrl-C or let it get killed — clips already generated and
downloads already complete are skipped on the next run, not redone.

## What it actually does, and the honest state of the 4 requested models

Read the top of `train_wakeword.py` — the docstring has the full breakdown,
but short version:

- **Qwen3-TTS-12Hz-1.7B-CustomVoice** and **Audio8-TTS-Preview-0.1b** are
  real, working, local TTS models and are what actually generates your
  positive clips. Both are used with `dtype=torch.float16`, not the
  `bfloat16` their model cards default to — a GTX 1080 (Pascal, sm_61) has
  no native bf16 support, and no flash-attention-2 either, so that's
  skipped too.
- **darkps/ice-012-audio** isn't released — the model card says "COMING
  SOON," no weights exist. The script checks the repo automatically on
  every run; if DarkPs publishes it later, you'll need to add a `load`/
  `generate` pair for it in `train_wakeword.py` (there's a `GeneratorSlot`
  stub already there — see `ICE012_SLOT`) since no working example exists
  yet to write one against.
- **higgs-audio-v3-tts-4b** is real, but its documented usage is a
  dedicated SGLang-Omni or vLLM-Omni *server* (Docker, `--gpus all`),
  benchmarked by Boson AI on an H100 — not a "load it and call generate()"
  local model. It's disabled by default. If you want it anyway, stand up
  your own server first:

  ```bash
  docker pull lmsysorg/sglang-omni:dev
  docker run -it --gpus all --shm-size 32g --ipc host --network host \
      lmsysorg/sglang-omni:dev /bin/zsh
  # inside the container:
  git clone git@github.com:sgl-project/sglang-omni.git && cd sglang-omni
  uv venv .venv -p 3.12 && source .venv/bin/activate
  uv pip install -v -e .
  export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
  hf download bosonai/higgs-audio-v3-tts-4b
  sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000
  ```

  then run this script with `--enable-higgs --higgs-url http://localhost:8000`.
  Expect it to be slow, or to not fit at all, on a GTX 1080 — this genuinely
  wants an H100-class card. Its license is also non-commercial-only
  (Boson Higgs Audio v3 Research and Non-Commercial License).

Negative training data is the same ~2,000-hour ACAV100M feature set
openWakeWord's own maintainer uses for the official pretrained models
(17.3 GB, downloaded automatically in the background from
`davidscripka/openwakeword_features` on Hugging Face — no TTS or GPU
involved in producing it, it's precomputed) plus an 11-hour validation set
for false-positive-rate tracking during training.

**This is a simplified version of openWakeWord's full training pipeline** —
no room-impulse-response reverb mixing, no background-noise mixing, no
adversarial-negative-phrase augmentation. Good enough to get a working
model; if false-positive rate in real use turns out too high, that
augmentation is the next thing to add (see
[`automatic_model_training.ipynb`](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb)
for the full version).

## After it finishes

The script prints (and logs) the exact config line to paste into
`~/.config/nixorb/config.toml`:

```toml
wake_word_model = "/home/you/.local/share/nixorb/wakeword_training/models/hey_nixorb.onnx,/home/you/.local/share/nixorb/wakeword_training/models/nixorb.onnx,/home/you/.local/share/nixorb/wakeword_training/models/hypernix.onnx"
```

(nixorb's `wake_word.py` now supports comma-separated model paths — any one
of them firing activates the orb.)

**You still need openwakeword actually installed to use the result** — see
the `wakeword` extra in nixorb's own `pyproject.toml`. It now installs from
GitHub main rather than PyPI, which fixes the `tflite-runtime`/Python 3.12
wall from earlier in this project's setup.
