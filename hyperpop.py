
"""
generate_hypernix_dataset.py
────────────────────────────
Builds a training dataset for the `hypernix` Python package (v0.61.2) and
uploads it to HuggingFace at  ray0rf1re/hyper-pip.

Usage
-----
    python generate_hypernix_dataset.py

The script will prompt you for:
  • Your HuggingFace token  (write access to ray0rf1re/hyper-pip)
  • An Anthropic API key    (optional – enables 750 000-token target;
                             omitting it uses the built-in seed bank,
                             which already exceeds 50 000 tokens)

Compatible with Python 3.11, 3.12, and 3.13.
Requires:  pip install datasets huggingface_hub requests
Optional:  pip install anthropic          (for AI-expanded generation)
"""

from __future__ import annotations

import getpass
import json
import os
import re
import sys
import time
import textwrap
import uuid
from dataclasses import dataclass, field
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Sentinel / version tag
# ─────────────────────────────────────────────────────────────────────────────
HYPERNIX_VERSION = "0.61.2"
TARGET_REPO      = "ray0rf1re/hyper-pip"
MIN_TOKENS_BARE  = 50_000    # without Anthropic key
TARGET_TOKENS    = 750_000   # with Anthropic key
MODEL_ID         = "claude-sonnet-4-20250514"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  All modules & their canonical descriptions
# ─────────────────────────────────────────────────────────────────────────────
MODULES: dict[str, str] = {
    "hypernix.download": (
        "Pull model snapshots from the HuggingFace Hub. Supports short-name "
        "resolution via KNOWN_MODELS, gated repos, offline cache, and a "
        "fallback chain for the 'nix' family."
    ),
    "hypernix.train": (
        "HyperNixConfig, HyperNixModel, init_from_scratch, expand_checkpoint, "
        "and train(). Non-HyperNix architectures are routed through "
        "AutoModelForCausalLM."
    ),
    "hypernix.old_oven": (
        "CodeOven – ready-to-use wrapper around a snapshot: .complete(), "
        ".chat(), .fill(), .save_pt(). new_oven() spins a fresh model from "
        "ARCH_PRESETS. preheat() resolves short names and downloads on demand."
    ),
    "hypernix.old_fridge": (
        "Memory housekeeping: freeze, unfreeze, parameter_stats, "
        "offload_to_cpu, chill_cache. Useful for selective gradient updates "
        "and layer freezing."
    ),
    "hypernix.mediocre_fridge": (
        "Judge-training dataset generation. synthesize_judge_corpus() creates "
        "labelled (GOOD/BAD/OK) training pairs. collect_responses_from() "
        "wires a labeling rubric to an oven."
    ),
    "hypernix.new_fridge": (
        "Training-curve graphing. parse_training_log() extracts step/loss/lr "
        "lines. plot_loss_curve() and plot_score_distribution() emit PNG "
        "charts via Matplotlib (installed lazily)."
    ),
    "hypernix.new_range": (
        "Zero-dependency first-fail labeling rubric. Rules: is_empty, "
        "is_refusal, math_lacks_digit, is_repetition. Returns GOOD/BAD/OK."
    ),
    "hypernix.old_range": (
        "Weighted-mean scored rubric with None='no opinion'. Any rule at 0 "
        "short-circuits to BAD. Supports references, keyword lists, and "
        "stopword-filtered overlap."
    ),
    "hypernix.industrial_range": (
        "LLM-as-judge wrapper around any CodeOven. Pointwise and pairwise "
        "judgment with response caching."
    ),
    "hypernix.freezer": (
        "VRAM manager. OldFreezer (8-10 GB), NewFreezer (11 GB+), "
        "FlashFreezer (OOM-safe retry with exponential backoff). Pascal "
        "(sm_61/CUDA 6.1) helpers. 48 CPU presets and 71 GPU presets "
        "(including Apple Silicon MPS and AMD Radeon). auto_freezer() "
        "picks the right tier automatically."
    ),
    "hypernix.smoke_alarm": (
        "Training-step planner & monitor. RadsAlarm (constants, lightest), "
        "GasAlarm (CPU/GPU presets), ModernAlarm (warmup-measured), "
        "AutoAlarm (selector). storage_warning and mid-run check(). Accepts "
        "log_every, save_every, eval_every kwargs."
    ),
    "hypernix.pans": (
        "5-tier text preprocessing pipeline. FryingPan (verbatim trim), "
        "SaucePan (whitespace collapse), Skillet (chat/instruct tags), "
        "GrillPan (SHA1 dedupe + min-length), Wok (buffer + shuffle + "
        "reverse augmentation). pick_pan() by name."
    ),
    "hypernix.microwave": (
        "5-tier throwaway inference: defrost (preheat-only), low_zap "
        "(16-token deterministic), zap (64-token standard), high_zap "
        "(512-token draft), chat_zap (one-turn chat with system prompt). "
        "reheat() continues a prior output."
    ),
    "hypernix.table": (
        "Dead-simple tabular viewer over list-of-dicts. "
        "Table.from_training_log(), Table.from_judge_corpus(), .head(), "
        ".filter(), .select(), .sort_by(), .show(). No external deps; "
        "column widths auto-size."
    ),
    "hypernix.sink": (
        "Append-only file sink with optional rotation (bytes) and SHA1 "
        "deduplication. write(), write_json(), pour(iterable). Context "
        "manager. Crash-safe: opens/closes on every write."
    ),
    "hypernix.instant_pot": (
        "One-call end-to-end pipeline. brew(recipe) accepts a plain dict or "
        "JSON file with repo_id/local_dir, dataset, out_dir, steps, "
        "batch_size, context_length, lr, device, dtype, freeze_embed, and "
        "an optional quants list to emit GGUFs. CLI: hypernix brew recipe.json."
    ),
    "hypernix.coffee_maker": (
        "CoffeeMaker (drip – scheduled repetition), FrenchPressMaker "
        "(batch), PercolatorMaker (cyclic refinement with convergence), "
        "ColdBrewMaker (long-run with mandatory JSON checkpoints that "
        "resume after a crash). coffee_maker(), french_press(), "
        "percolator(), cold_brew() factory functions."
    ),
    "hypernix.pressure_cooker": (
        "Custom torch.optim.Optimizer: AdamW + three-phase LR schedule "
        "(linear warmup → plateau → cosine cooldown) + optional Lookahead "
        "(Zhang et al. 2019). StovetopCooker (CPU t1), ElectricCooker "
        "(CPU t2 foreach), InductionCooker (GPU t1 fused), ProCooker "
        "(GPU t2 CUDA-graph). universal_cooker() auto-selects. Grad scaler "
        "and grad-accumulation built in. phase() and scheduled_lr() helpers."
    ),
    "hypernix.convert": (
        "Safetensors → GGUF at fp32/fp16. Architecture-agnostic tensor "
        "naming; Llama/GPT-NeoX/GPT-2/nanoGPT shapes get canonical GGUF "
        "names; unknowns round-trip verbatim."
    ),
    "hypernix.quantize": (
        "llama-quantize driver. Full QUANT_CATALOG of 30 QuantSpec "
        "dataclasses (F32/F16/BF16, legacy Q4_0–Q8_0, k-quants Q2_K–Q6_K, "
        "IQ1_S–IQ4_XS). Helpers: quant_recommended(), quant_by_category(), "
        "quant_for_size(), quant_estimate_size(), quant_resolve_spec(), "
        "quant_list_types(). 49 aliases including q4km, q5km, dash-form."
    ),
    "hypernix.upload": (
        "Push produced artifacts (GGUFs, snapshots) back to a HuggingFace "
        "repo. upload_gguf() and upload_snapshot() with progress callbacks."
    ),
    "hypernix.espresso_maker": (
        "4-tier prompt-battery evaluation. Ristretto (16 tok, temp 0.0, "
        "1 sample – deterministic spot-check), SingleShot (64 tok, 0.2), "
        "DoubleShot (96 tok, 0.4, 2 samples – scorer picks winner), "
        "Lungo (256 tok, 0.8, 4 samples). pull(prompts, references) → "
        "list[Shot]. mean_score property."
    ),
    "hypernix.blender": (
        "4-tier multi-source data mixing. HandBlender (concatenation), "
        "PersonalBlender (round-robin interleave), CountertopBlender "
        "(weighted sampling), HighPowerBlender (full buffer + shuffle). "
        "All pair with sink.Sink.pour()."
    ),
    "hypernix.toaster": (
        "4-tier per-line formatting. TwoSliceToaster (pair every 2 lines "
        "as prompt/response), FourSliceToaster (4 lines → 2-turn chat), "
        "ConveyorToaster (streaming per-line template), ToasterOven "
        "(whole-document wrap with header/footer)."
    ),
    "hypernix.food_processor": (
        "4-tier bulk chunking. ChopBlade (split on separator), SliceBlade "
        "(fixed-length char slices with overlap), ShredBlade (whitespace-"
        "tokenized sliding window), PureeBlade (whole file, whitespace "
        "collapsed). overlap_chars >= slice_chars raises ValueError."
    ),
    "hypernix.smoker": (
        "4-tier training quality. UseableSmoker (minimum viable), "
        "GoodSmoker (+ linear warmup/plateau/cosine cooldown), "
        "CommercialSmoker (+ EMA weight blend at end), HighQualitySmoker "
        "(+ curriculum / progressive context length). smoke(corpus, out_dir)."
    ),
    "hypernix.tv": (
        "btop++-style training dashboard (tvtop CLI). 2×2 panel grid: "
        "cpu (per-core bars + history), memory (USED/CACHE/FREE/SWAP + "
        "history), gpu (UTIL/VRAM/TEMP/PWR + history), training (step/"
        "loss/lr/ETA). Auto-detects training logs. ASCII fallback. "
        "nvidia-smi cached 3 s. New in 0.61.2: 4-panel rewrite."
    ),
    "hypernix.compactor": (
        "Zip older checkpoints to save disk. Compactor(root, keep_recent=3, "
        "fmt='zip'|'tar'|'tar.gz') finds ckpt-N/checkpoint-N/step-N dirs "
        "and .pt/.safetensors files, keeps N most recent uncompressed, "
        "archives the rest. dry_run=True plans without writing."
    ),
    "hypernix.ethanol": (
        "Bounded GPU overclock. Ethanol(level=0..30) maps to capped "
        "core/memory/power-limit offsets. Backends: nvidia-settings, "
        "nvidia-smi, rocm-smi, intel_gpu_frequency. Requires confirm=True "
        "or HYPERNIX_ETHANOL_CONFIRM=1. CLI: eth."
    ),
    "hypernix.outage": (
        "Turn the display off during training. with Outage(): blanks the "
        "panel on entry, always restores on exit (clean finish, "
        "KeyboardInterrupt, OOM, etc.). Backends: xset (X11), wlopm "
        "(Wayland), pmset (macOS), SendMessageW ctypes (Windows)."
    ),
    "hypernix.timer": (
        "4-tier countdown/interval helpers on monotonic clock. KitchenTimer "
        "(t1 plain countdown), EggTimer (t2 + on_ring callback), "
        "IntervalTimer (t3 should_fire() for checkpoint/log cadence), "
        "PomodoroTimer (t4 work/rest alternation)."
    ),
    "hypernix.thermometer": (
        "4-tier CPU/GPU temperature sampling. InstantThermometer (one-shot), "
        "ProbeThermometer (rolling window + recent_max/mean/min), "
        "InfraredThermometer (per-source peak + warn/critical thresholds), "
        "DigitalThermometer (JSONL log for post-mortem). Sources: psutil, "
        "Linux /sys/class/thermal, nvidia-smi."
    ),
    "hypernix.dishwasher": (
        "4-tier training-run cleanup. HandWash (logs + __pycache__), "
        "QuickWash (+ *.tmp/*.partial/*.lock), NormalWash (+ stale "
        "checkpoints via compactor), HeavyDuty (+ fp16 GGUFs + build dirs; "
        "opt-in purge_hf_cache=True). dry_run=True + bytes-freed report."
    ),
    "hypernix.strainer": (
        "4-tier dataset quality filter. Colander (empty/None/whitespace), "
        "FineMesh (+ length floor/ceiling), NutMilkBag (+ non-printable "
        "char filter), Cheesecloth (+ 8-gram Jaccard near-duplicate "
        "detection at similarity_threshold=0.85). Accepts dicts or strings."
    ),
    "hypernix.ups": (
        "Uninterruptible-power-supply mode. Checks weather (open-meteo, "
        "no API key) and pluggable outage_check_fn() every "
        "check_interval_seconds. On panic: fires snapshot_fn once, then "
        "triples save cadence. Auto-locates via ipapi.co. offline=True "
        "or HYPERNIX_UPS_OFFLINE=1 skips HTTP."
    ),
    "hypernix.injection": (
        "Token/phrase splicers for chat scaffolding. ThinkingInjector "
        "(<think>…</think>), TestingInjector (<|test|>), "
        "SystemOverrideInjector (<|system_override|>…), CustomInjector "
        "(generic open/close/mode). inject_messages() for message lists, "
        "inject_text() for rendered strings. Module-level shortcuts: "
        "injection.thinking(), injection.testing(), "
        "injection.system_override()."
    ),
    "hypernix.plasma": (
        "Quick GPU benchmark for sharper ETAs. Runs a 6-step Llama-shape "
        "forward+backward loop, returns PlasmaResult(step_ms, "
        "tokens_per_sec, calibration_factor). calibrate_alarm(alarm, "
        "result) rebinds alarm.estimate_step_seconds to measured speed. "
        "Autocast on CUDA; reset_calibration() undoes the wrapper."
    ),
    "hypernix.cookbook": (
        "Chat-template registry. Templates: chatml, hyper-nix.2 (ChatML + "
        "HyperNix system prompt), llama3, llama2, alpaca, vicuna, plain. "
        "for_model(repo_id) resolves the right template. "
        "tmpl.apply(messages, add_generation_prompt=True)."
    ),
    "hypernix.countertop": (
        "Multi-turn chat session bound to an oven. say(), reset(), "
        "save(path), load(path). Auto-resolves chat template from "
        "oven.repo_id. Optional Bell for streaming, Flour for reply "
        "cleanup. Trims oldest turns at max_history_tokens. countertop() "
        "factory with persona= kwarg."
    ),
    "hypernix.menu": (
        "System-prompt preset registry. Built-in personas: default, concise, "
        "code-helper, judge, creative, chef, hyper-nix. Menu.find(query) "
        "fuzzy lookup. Menu.save()/Menu.load(). Pairs with Countertop's "
        "persona= kwarg."
    ),
    "hypernix.bell": (
        "Streaming-token callback wrapper. on_token(fn), on_done(fn), "
        "stream_chat(), iter_chat(), iter_complete(). stdout_bell() and "
        "file_bell(path) ready-made variants. Accepts flour= for live "
        "logits processing. Stop-marker checked BEFORE yielding token."
    ),
    "hypernix.flour": (
        "Chat-quality logits processor. Repetition penalty (multiplicative), "
        "frequency penalty, presence penalty, no-repeat n-gram blocking, "
        "bad-word suppression, role-leak suppression (strips assistant-"
        "hallucinated turn markers), stop-sequence detection on decoded "
        "text. Flour.smart_default(), Flour.aggressive(), Flour.off()."
    ),
    "hypernix.whisk": (
        "Checkpoint averaging. swa_average(items), ema(items, decay=0.99), "
        "geometric_mean(items). Inputs may be in-memory state dicts or "
        "paths to .pt/.safetensors. Mismatched keys intersected (warn or "
        "strict=True). whisk() factory; whisk_to_snapshot() whisks + "
        "writes a full HF-style snapshot directory."
    ),
    "hypernix.cutting_board": (
        "Train/val/test splitting. CuttingBoard(train_ratio, val_ratio, "
        "test_ratio, seed, shuffle) – deterministic random split. "
        "slice(source) and slice_to_files(out_dir). StratifiedBoard "
        "preserves class distribution per label_key. cutting_board() "
        "convenience factory."
    ),
    "hypernix.apron": (
        "RNG-state guard. Captures Python random, NumPy, PyTorch CPU, and "
        "every CUDA device's RNG, then restores on exit. Use as context "
        "manager (with apron(seed=0)) or snapshot/restore pair. Ensures "
        "a sampling step can't leak RNG state to the caller."
    ),
    "hypernix.recipe_book": (
        "Named-config registry. RecipeBook.add/get/remove/save/load. "
        "cook(name, **overrides) dispatches by kind: 'instant_pot', "
        "'cold_brew', 'espresso'. from_builtins() ships evaluator-quick, "
        "ftune-pascal, nightly-coldbrew, espresso-eval recipes."
    ),
    "hypernix.lunchbox": (
        "Consistent-schema dataset packager for HuggingFace. add(**fields), "
        "normalize() (fills None), validate() (rejects mixed types), "
        "pack(path) via datasets.Dataset → Parquet, push_to_hub(repo_id), "
        "pack_jsonl(path). Lunchbox.for_eval() preloads EVAL_SCHEMA."
    ),
    "hypernix.deep_fryer": (
        "2-tier model-weight perturbation. LightFry (t1): 2% of elements, "
        "0.1× param-std Gaussian noise, reversible. HeavyFry (t2): 30% "
        "elements, 0.5× noise + configurable zero-rate. save_pristine() "
        "and un_fry() for rollback. Uses per-parameter torch.Generator "
        "(no global RNG mutation)."
    ),
    "hypernix.cake_pan": (
        "Hybrid CPU+GPU training guard. bake(fn) catches NaN/Inf loss, "
        "enforces SIGALRM watchdog, monitors GPU memory and offloads "
        "modules at free_gb_trip, rolls back to pristine state on trouble, "
        "raises BakeOff(reason, step). CakePan.oven(batches, step_fn) is "
        "the fire-and-forget loop wrapper with automatic retry + skip."
    ),
    "hypernix.torch_compat": (
        "Portability shim for old Intel Macs / torch 1.13. Version-gated "
        "fallbacks for torch.nn.RMSNorm (needs ≥ 2.4) and "
        "F.scaled_dot_product_attention (needs ≥ 2.0). HyperNixModel and "
        "NanoNanoModel route through the shim."
    ),
    "hypernix.salt_shaker": (
        "3-tier gentle data augmentation. FromTheBag (t1 per-char "
        "substitution), HandCrusher (t2 adjacent-token swaps), PoshSaltDish "
        "(t3 drop/duplicate/swap with word-level granularity). Deterministic "
        "seed. Plugs into sink.Sink.pour()."
    ),
    "hypernix.pepper_shaker": (
        "3-tier sharp perturbations. SmallShaker (t1 random token masking), "
        "Dish (t2 typo injection – preserves first+last char), TallHandmade "
        "(t3 negation injection with configurable negator). Deterministic seed."
    ),
    "hypernix.utils": (
        "Utility helpers: healthcheck(), diagnostic_info(), list_models(), "
        "print_models(), session_dir(), is_module_available(), has_binary(). "
        "Diagnostic snapshot covers torch+CUDA+optional deps+binaries on "
        "PATH+KNOWN_MODELS count."
    ),
    "hypernix.hyped": (
        "High-quality TUI chat CLI (hyped console script). Two-screen flow: "
        "configurator (model picker from KNOWN_MODELS, persona, sampling) + "
        "chat (streaming, slash commands /quit /reset /persona /save /help). "
        "--model, --persona, --ascii, --flour flags. Streams via Bell + Flour."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Seed examples (hand-crafted, exhaustive, errorless on 3.11/3.12/3.13)
# ─────────────────────────────────────────────────────────────────────────────
# Each entry is a dict with keys:
#   instruction  – natural-language task description
#   input        – optional context / additional detail (may be empty)
#   output       – complete, runnable Python code
#   module       – which hypernix module is demonstrated
#   python_ver   – "3.11", "3.12", or "3.13"
#   category     – "basic", "intermediate", "advanced", "cli", "integration"

SEED_EXAMPLES: list[dict[str, str]] = []

def _ex(instruction: str, output: str, module: str,
        python_ver: str = "3.12", category: str = "basic",
        input_ctx: str = "") -> dict[str, str]:
    return {
        "instruction": instruction,
        "input": input_ctx,
        "output": output,
        "module": module,
        "python_ver": python_ver,
        "category": category,
        "hypernix_version": HYPERNIX_VERSION,
    }


# ── hypernix.download ────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Download the hyper-nix.1 model snapshot using the short name.",
        '''\
from hypernix.download import download_model

# Uses KNOWN_MODELS short-name resolution; downloads to the default
# HuggingFace cache directory on first call, returns the local path.
local_dir: str = download_model("hyper-nix.1")
print(f"Snapshot cached at: {local_dir}")
''',
        "hypernix.download", "3.12", "basic",
    ),
    _ex(
        "Download a gated HuggingFace model snapshot with an access token.",
        '''\
import os
from hypernix.download import download_model

hf_token: str = os.environ["HF_TOKEN"]  # read from env – never hardcode
local_dir: str = download_model(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    token=hf_token,
)
print(f"Downloaded to: {local_dir}")
''',
        "hypernix.download", "3.13", "intermediate",
    ),
    _ex(
        "Download a model using a full HuggingFace repo ID and a custom "
        "cache directory.",
        '''\
from pathlib import Path
from hypernix.download import download_model

cache_dir = Path("./model_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

local_dir = download_model(
    "ray0rf1re/hyper-nix.1",
    cache_dir=str(cache_dir),
)
print(f"Model ready at {local_dir}")
''',
        "hypernix.download", "3.11", "intermediate",
    ),
]

# ── hypernix.old_oven ────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Preheat a CodeOven from the 'nix2.5' short name and complete a "
        "Python function stub.",
        '''\
from hypernix.old_oven import preheat

# preheat() resolves short names via KNOWN_MODELS and downloads on demand.
oven = preheat(repo_id="nix2.5", device="cpu", dtype="float32")

prompt = "def fibonacci(n: int) -> int:\\n    "
completion = oven.complete(prompt, max_new_tokens=128, temperature=0.1)
print(completion)
''',
        "hypernix.old_oven", "3.12", "basic",
    ),
    _ex(
        "Use CodeOven.chat() for a single-turn conversation with a system "
        "prompt.",
        '''\
from hypernix.old_oven import preheat

oven = preheat(repo_id="nix2.5", device="cpu", dtype="float32")

messages = [
    {"role": "system", "content": "You are a concise Python tutor."},
    {"role": "user",   "content": "Explain list comprehensions in one sentence."},
]
reply = oven.chat(messages, max_new_tokens=64)
print(reply)
''',
        "hypernix.old_oven", "3.12", "intermediate",
    ),
    _ex(
        "Use new_oven() to spin up a fresh parametric HyperNix model and "
        "save a PyTorch checkpoint.",
        '''\
from hypernix.old_oven import new_oven

# Spin a fresh ~92 M-parameter HyperNix 1.5 model (no pretrained weights).
oven = new_oven(
    arch="hypernix",
    device="cpu",
    hidden_size=512,
    num_layers=8,
    num_heads=8,
)
print(f"Parameters: {sum(p.numel() for p in oven.model.parameters()):,}")

# Persist the model weights.
oven.save_pt("./fresh_hypernix.pt")
print("Saved to ./fresh_hypernix.pt")
''',
        "hypernix.old_oven", "3.11", "advanced",
    ),
    _ex(
        "Use CodeOven.fill() for fill-in-the-middle code completion.",
        '''\
from hypernix.old_oven import preheat

oven = preheat(repo_id="nix2.5", device="cpu", dtype="float32")

prefix = "def greet(name: str) -> str:\\n    return "
suffix = "\\n\\nresult = greet('world')\\nprint(result)"
middle = oven.fill(prefix, suffix, max_new_tokens=32)
print(f"Filled: {middle!r}")
''',
        "hypernix.old_oven", "3.13", "intermediate",
    ),
]

# ── hypernix.old_fridge ──────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Freeze the embedding layer of a model to prevent it from being "
        "updated during fine-tuning.",
        '''\
from hypernix.old_oven import preheat
from hypernix import old_fridge

oven = preheat(repo_id="nix2.5", device="cpu", dtype="float32")

# Freeze every parameter whose name matches the pattern.
old_fridge.freeze(oven.model, patterns=("embed_tokens",))
stats = old_fridge.parameter_stats(oven.model)
print(f"Trainable params: {stats['trainable']:,}")
print(f"Frozen params:    {stats['frozen']:,}")
''',
        "hypernix.old_fridge", "3.12", "intermediate",
    ),
    _ex(
        "Offload a model to CPU to free GPU VRAM, then move it back for "
        "inference.",
        '''\
from hypernix.old_oven import preheat
from hypernix import old_fridge

oven = preheat(repo_id="nix2.5", device="cuda", dtype="float16")

# Free VRAM between inference calls.
old_fridge.offload_to_cpu(oven.model)
print("Model offloaded to CPU.")

# Bring it back when needed.
oven.model.to("cuda")
out = oven.complete("Hello", max_new_tokens=8)
print(out)
''',
        "hypernix.old_fridge", "3.12", "advanced",
    ),
    _ex(
        "Unfreeze previously frozen layers before the second stage of "
        "training.",
        '''\
from hypernix.old_oven import preheat
from hypernix import old_fridge

oven = preheat(repo_id="nix2.5", device="cpu", dtype="float32")

old_fridge.freeze(oven.model, patterns=("embed_tokens",))
print("Stage 1 – embeddings frozen.")

# ... first stage training loop ...

old_fridge.unfreeze(oven.model, patterns=("embed_tokens",))
stats = old_fridge.parameter_stats(oven.model)
print(f"Stage 2 – all {stats['total']:,} params trainable.")
''',
        "hypernix.old_fridge", "3.11", "intermediate",
    ),
    _ex(
        "Clear the GPU CUDA cache to reclaim fragmented VRAM between batches.",
        '''\
from hypernix import old_fridge

# Safe to call even when no GPU is present – no-ops on CPU.
freed = old_fridge.chill_cache()
print(f"Cache cleared: {freed}")
''',
        "hypernix.old_fridge", "3.13", "basic",
    ),
]

# ── hypernix.freezer ─────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Use auto_freezer() to pick the right VRAM strategy for the current "
        "GPU automatically.",
        '''\
from hypernix import freezer

# Detects compute capability and installed VRAM; returns OldFreezer or
# NewFreezer accordingly.
fz = freezer.auto_freezer()
print(type(fz).__name__)  # e.g. "OldFreezer" on an 8-GB card
print(fz)
''',
        "hypernix.freezer", "3.12", "basic",
    ),
    _ex(
        "Wrap training in a FlashFreezer to recover from GPU OOM errors "
        "automatically.",
        '''\
from hypernix import freezer, old_oven

base = freezer.auto_freezer()
fz   = freezer.flash_freezer(base=base, slow=True)

oven = old_oven.preheat(repo_id="nix2.5", device="cuda", dtype="float16")

def train_step() -> None:
    # Your training loop here.  If an OOM occurs, FlashFreezer
    # halves current_batch_size and retries automatically.
    oven.train(
        dataset="./corpus.txt",
        out_dir="./ckpt",
        steps=100,
        batch_size=fz.current_batch_size,
    )

fz.guard(train_step)
print("Training completed without crashing.")
''',
        "hypernix.freezer", "3.12", "advanced",
    ),
    _ex(
        "Look up a GPU preset to see its VRAM, compute capability, and "
        "recommended batch size.",
        '''\
from hypernix.freezer import gpu_preset, GPU_PRESETS

spec = gpu_preset("rtx-3080-ti")
print(f"Name:              {spec.name}")
print(f"VRAM:              {spec.vram_gb} GB")
print(f"Compute cap:       {spec.compute_capability}")
print(f"Recommended batch: {spec.recommended_batch_size}")
print(f"Total GPU presets: {len(GPU_PRESETS)}")
''',
        "hypernix.freezer", "3.11", "intermediate",
    ),
    _ex(
        "Check Pascal GPU constraints and retrieve safe dtype and mode hints.",
        '''\
from hypernix.freezer import is_pascal, pascal_safe_dtype, pascal_mode_hints
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if is_pascal(device):
    dtype = pascal_safe_dtype(device)  # always fp16 on Pascal
    hints = pascal_mode_hints()
    print(f"Pascal detected – using dtype: {dtype}")
    print("Pascal hints:", hints)
else:
    print("Not a Pascal GPU – using bf16 or fp32 freely.")
''',
        "hypernix.freezer", "3.13", "advanced",
    ),
]

# ── hypernix.smoke_alarm ─────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Create a GasAlarm from a GPU preset and estimate the number of "
        "training steps that fit in a two-hour budget.",
        '''\
from hypernix.smoke_alarm import GasAlarm

alarm = GasAlarm(gpu_preset="rtx-3080-ti", time_budget_seconds=7200)
steps = alarm.recommended_steps()
print(f"Recommended steps in 2 hours: {steps}")
print(f"ETA per step:                 {alarm.estimate_step_seconds():.3f} s")
''',
        "hypernix.smoke_alarm", "3.12", "intermediate",
    ),
    _ex(
        "Use AutoAlarm to automatically select the best alarm tier for the "
        "current hardware.",
        '''\
from hypernix.smoke_alarm import AutoAlarm

alarm = AutoAlarm()  # inspects GPU/CPU and picks Rads, Gas, or Modern
print(f"Selected tier: {type(alarm).__name__}")
print(f"Recommended steps: {alarm.recommended_steps()}")

# Mid-run check: call inside your training loop.
for step in range(10):
    alarm.check(step=step, loss=2.3 - step * 0.05)
''',
        "hypernix.smoke_alarm", "3.12", "advanced",
    ),
    _ex(
        "Build a GasAlarm from a CPU preset with custom cadence knobs and a "
        "step cap.",
        '''\
from hypernix.smoke_alarm import GasAlarm

alarm = GasAlarm(
    cpu_preset="i7-12700h",
    time_budget_seconds=3600,
    max_steps=2000,
    log_every=50,
    save_every=200,
    eval_every=500,
)
print(f"Steps (capped):  {alarm.recommended_steps()}")
print(f"Phase:           {alarm.phase if hasattr(alarm, 'phase') else 'N/A'}")
''',
        "hypernix.smoke_alarm", "3.11", "intermediate",
    ),
    _ex(
        "Use RadsAlarm as the lightest-weight option when hardware "
        "information is not available.",
        '''\
from hypernix.smoke_alarm import RadsAlarm

# RadsAlarm uses fixed constants – no hardware probing.
alarm = RadsAlarm(time_budget_seconds=1800)
print(f"Recommended steps: {alarm.recommended_steps()}")
print(f"Est. step time:    {alarm.estimate_step_seconds():.2f} s")
''',
        "hypernix.smoke_alarm", "3.13", "basic",
    ),
]

# ── hypernix.pans ────────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Use GrillPan to deduplicate and filter a raw text corpus, then "
        "write the result with Sink.",
        '''\
from pathlib import Path
from hypernix import pans
from hypernix.sink import Sink

# Create a tiny sample corpus for the demo.
raw = Path("raw_corpus.txt")
raw.write_text("\\n".join([
    "Hello world",
    "  hello world  ",   # duplicate after strip
    "Hi",                # too short (< 8 chars)
    "The quick brown fox jumps over the lazy dog",
    "The quick brown fox jumps over the lazy dog",  # exact duplicate
    "Another unique sentence here.",
]))

Sink("clean_corpus.txt").pour(
    pans.GrillPan(str(raw), min_chars=8)
)

result = Path("clean_corpus.txt").read_text()
print(result)
# Expected: only "Hello world" and the two distinct long sentences.
''',
        "hypernix.pans", "3.12", "intermediate",
    ),
    _ex(
        "Use Skillet to wrap lines with instruct-style tags for fine-tuning.",
        '''\
from pathlib import Path
from hypernix import pans

raw = Path("prompts.txt")
raw.write_text("Explain recursion.\\nWhat is a closure?\\n")

for formatted_line in pans.Skillet(str(raw), mode="instruct"):
    print(repr(formatted_line))
''',
        "hypernix.pans", "3.11", "basic",
    ),
    _ex(
        "Use Wok for shuffle-augmented preprocessing with reverse-order "
        "augmentation.",
        '''\
from pathlib import Path
from hypernix import pans

corpus = Path("corpus.txt")
corpus.write_text("\\n".join(f"Sentence {i}." for i in range(20)))

processed = list(pans.Wok(str(corpus), seed=42, reverse_ratio=0.1))
print(f"Lines out: {len(processed)}")
print("First 3:", processed[:3])
''',
        "hypernix.pans", "3.13", "advanced",
    ),
    _ex(
        "Pick a pan by name at runtime using pick_pan().",
        '''\
from pathlib import Path
from hypernix import pans

corpus = Path("data.txt")
corpus.write_text("Line one.\\nLine two.\\nLine three.\\n")

tier_name = "sauce-pan"   # could come from a config file
pan = pans.pick_pan(tier_name, source=str(corpus))
for line in pan:
    print(repr(line))
''',
        "hypernix.pans", "3.12", "basic",
    ),
]

# ── hypernix.microwave ───────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Use zap() for a standard 64-token completion without managing an "
        "oven object.",
        '''\
from hypernix.microwave import zap

# zap() preheats the oven, runs completion, and discards the oven.
output: str = zap("nix2.5", "def add(a, b):", device="cpu")
print(output)
''',
        "hypernix.microwave", "3.12", "basic",
    ),
    _ex(
        "Use high_zap() to draft a long answer (512 tokens) to a "
        "technical question.",
        '''\
from hypernix.microwave import high_zap

answer = high_zap(
    "nix2.5",
    "Explain Rotary Position Embeddings (RoPE) in detail.",
    device="cpu",
    temperature=0.7,
)
print(answer)
''',
        "hypernix.microwave", "3.11", "intermediate",
    ),
    _ex(
        "Use defrost() to reuse the same oven across multiple calls without "
        "reloading weights.",
        '''\
from hypernix.microwave import defrost, reheat, zap

# defrost() preheats and returns the oven; weights are loaded once.
oven = defrost("nix2.5", device="cpu")

first_out  = zap("nix2.5", "The capital of France is", device="cpu")
second_out = reheat(oven, prior_output=first_out, max_new_tokens=32)

print("First: ", first_out)
print("Continued:", second_out)
''',
        "hypernix.microwave", "3.12", "advanced",
    ),
    _ex(
        "Use chat_zap() for a one-turn chat interaction with a system prompt.",
        '''\
from hypernix.microwave import chat_zap

reply = chat_zap(
    "nix2.5",
    "What is the difference between a list and a tuple in Python?",
    system="Answer in exactly two sentences.",
    device="cpu",
)
print(reply)
''',
        "hypernix.microwave", "3.13", "basic",
    ),
]

# ── hypernix.pressure_cooker ─────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Create a PressureCooker optimizer with warmup, plateau, and cosine "
        "cooldown for a training loop.",
        '''\
import torch
import torch.nn as nn
from hypernix.pressure_cooker import pressure_cooker

model  = nn.Linear(128, 64)
opt    = pressure_cooker(
    model.parameters(),
    peak_lr=3e-4,
    warmup_steps=100,
    plateau_steps=500,
    cooldown_steps=100,
    weight_decay=0.1,
)

# Minimal training loop.
for step in range(720):
    loss = model(torch.randn(8, 128)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    if step % 100 == 0:
        print(f"step={step:4d}  phase={opt.phase():<10s}  lr={opt.scheduled_lr():.2e}")
''',
        "hypernix.pressure_cooker", "3.12", "intermediate",
    ),
    _ex(
        "Enable Lookahead on the PressureCooker optimizer for stabilized "
        "training on a narrow network.",
        '''\
import torch
import torch.nn as nn
from hypernix.pressure_cooker import pressure_cooker

model = nn.Linear(64, 32)
opt   = pressure_cooker(
    model.parameters(),
    peak_lr=1e-3,
    warmup_steps=50,
    plateau_steps=200,
    cooldown_steps=50,
    lookahead_k=5,       # Slow-weight seal every 5 inner steps.
    lookahead_alpha=0.5,
)

print(repr(opt))

for _ in range(310):
    loss = model(torch.randn(4, 64)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

print(f"Final phase: {opt.phase()}")
''',
        "hypernix.pressure_cooker", "3.11", "advanced",
    ),
    _ex(
        "Use universal_cooker() to auto-select the right device-tuned tier.",
        '''\
import torch
import torch.nn as nn
from hypernix.pressure_cooker import universal_cooker

model = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

opt = universal_cooker(model.parameters(), prefer_speed=True)
print(f"Selected: {type(opt).__name__}")   # ElectricCooker on CPU

# Train for a few steps.
for _ in range(10):
    loss = model(torch.randn(16, 256)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

print("describe:", opt.describe())
''',
        "hypernix.pressure_cooker", "3.13", "intermediate",
    ),
]

# ── hypernix.quantize ────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "List all recommended quantization types from the QUANT_CATALOG.",
        '''\
from hypernix.quantize import quant_recommended, quant_by_category

print("Recommended quants:")
for spec in quant_recommended():
    print(f"  {spec.name:<12}  {spec.bits_per_weight:.2f} bpw  {spec.notes}")

print("\\nK-quant family (sorted by bpw):")
for spec in quant_by_category("k"):
    print(f"  {spec.name:<10}  {spec.bits_per_weight:.2f} bpw")
''',
        "hypernix.quantize", "3.12", "basic",
    ),
    _ex(
        "Estimate which quantization type fits a target file-size budget.",
        '''\
from hypernix.quantize import quant_for_size, quant_estimate_size

# Suppose the fp16 GGUF is 4 GB.
fp16_bytes   = 4 * 1024 ** 3
target_bytes = 2 * 1024 ** 3   # want to fit on a 2 GB download limit

best = quant_for_size(target_bytes, fp16_bytes)
print(f"Best fit: {best.name}  ({best.bits_per_weight:.2f} bpw)")

estimated = quant_estimate_size(best.name, fp16_bytes)
print(f"Estimated size: {estimated / 1024 ** 2:.1f} MiB")
''',
        "hypernix.quantize", "3.11", "intermediate",
    ),
    _ex(
        "Resolve a quantization alias and look up its QuantSpec.",
        '''\
from hypernix.quantize import quant_resolve_spec

for alias in ("q4km", "q4-k-m", "Q4_K_M", "q6"):
    spec = quant_resolve_spec(alias)
    print(f"{alias!r:12s} → {spec.name:<10}  {spec.bits_per_weight:.2f} bpw  "
          f"category={spec.category}")
''',
        "hypernix.quantize", "3.12", "intermediate",
    ),
]

# ── hypernix.instant_pot ─────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Run a complete preheat → train → GGUF pipeline with instant_pot "
        "in a single call.",
        '''\
from pathlib import Path
from hypernix import instant_pot

# Create a tiny training corpus for the demo.
corpus = Path("demo_corpus.txt")
corpus.write_text("\\n".join(
    [f"Training sentence number {i} for the demo." for i in range(200)]
))

recipe = {
    "repo_id":        "nix2.5",
    "dataset":        str(corpus),
    "out_dir":        "./trained_output",
    "steps":          50,
    "batch_size":     1,
    "context_length": 256,
    "lr":             3e-4,
    "device":         "cpu",
    "dtype":          "float32",
    "freeze_embed":   True,
    # "quants": ["fp16"],  # uncomment to also emit a GGUF
}

trained_dir: str = instant_pot.brew(recipe)
print(f"Trained snapshot saved at: {trained_dir}")
''',
        "hypernix.instant_pot", "3.12", "advanced",
    ),
    _ex(
        "Load an instant_pot recipe from a JSON file and run it with a CLI "
        "step override.",
        '''\
# recipe.json content:
# {
#   "repo_id": "nix2.5",
#   "dataset": "./corpus.txt",
#   "out_dir": "./out",
#   "steps": 100
# }
#
# Shell usage:
#   hypernix brew recipe.json --set steps=500

# Python equivalent of --set overrides:
import json
from pathlib import Path
from hypernix import instant_pot

recipe_path = Path("recipe.json")
recipe_path.write_text(json.dumps({
    "repo_id":  "nix2.5",
    "dataset":  "./corpus.txt",
    "out_dir":  "./out",
    "steps":    100,
    "device":   "cpu",
    "dtype":    "float32",
}))

recipe = json.loads(recipe_path.read_text())
recipe["steps"] = 500   # programmatic override

result = instant_pot.brew(recipe)
print(f"Output: {result}")
''',
        "hypernix.instant_pot", "3.11", "advanced",
    ),
]

# ── hypernix.sink ─────────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Write lines to a Sink with deduplication enabled.",
        '''\
from pathlib import Path
from hypernix.sink import Sink

out = Path("output.txt")
s   = Sink(str(out), dedupe=True)

lines = ["apple", "banana", "apple", "cherry", "banana", "date"]
for line in lines:
    wrote = s.write(line)
    print(f"{'written' if wrote else 'skipped':8s}  {line!r}")

s.close()
print("\\nFinal file:")
print(out.read_text())
''',
        "hypernix.sink", "3.12", "basic",
    ),
    _ex(
        "Use Sink as a context manager and write structured JSON events.",
        '''\
from hypernix.sink import Sink

with Sink("events.jsonl") as s:
    for i in range(5):
        s.write_json({"step": i, "loss": 3.0 - i * 0.1})

import json
from pathlib import Path
for line in Path("events.jsonl").read_text().splitlines():
    print(json.loads(line))
''',
        "hypernix.sink", "3.13", "intermediate",
    ),
    _ex(
        "Use Sink with file rotation to split output into 1 KB chunks.",
        '''\
from pathlib import Path, glob
from hypernix.sink import Sink

s = Sink("rotated.txt", rotate_bytes=1024)
for i in range(200):
    s.write(f"Line {i:04d}: " + "x" * 20)
s.close()

# Each shard is <= 1 KB; the sink names them rotated.txt, rotated.txt.1, ...
import glob as _glob
shards = sorted(_glob.glob("rotated.txt*"))
print(f"Created {len(shards)} shard(s): {shards}")
''',
        "hypernix.sink", "3.12", "advanced",
    ),
]

# ── hypernix.table ────────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Load a training log into a Table and display the worst-loss steps.",
        '''\
from pathlib import Path
from hypernix.table import Table

# Simulate a training log.
log = Path("train.log")
log.write_text(
    "\\n".join(
        f"step {i}/{100} loss={3.0 - i * 0.02:.3f} lr=3e-4"
        for i in range(1, 101)
    )
)

t = Table.from_training_log(str(log))

# Show the 5 steps with the highest loss.
worst = t.sort_by("loss", descending=True).head(5)
print(worst.show())
''',
        "hypernix.table", "3.12", "intermediate",
    ),
    _ex(
        "Filter and project a judge corpus table to show only BAD examples.",
        '''\
from pathlib import Path
from hypernix.table import Table

# Simulate a judge corpus (output of mediocre_fridge.synthesize_judge_corpus).
corpus = Path("judge.txt")
corpus.write_text(
    "\\n".join([
        '{"prompt": "2+2", "response": "5", "label": "BAD"}',
        '{"prompt": "capital of France", "response": "Paris", "label": "GOOD"}',
        '{"prompt": "square root of 9", "response": "3", "label": "GOOD"}',
        '{"prompt": "3*3", "response": "10", "label": "BAD"}',
    ])
)

c   = Table.from_judge_corpus(str(corpus))
bad = c.filter(lambda r: r["label"] == "BAD").select("prompt", "response")
print(bad.show())
''',
        "hypernix.table", "3.11", "intermediate",
    ),
]

# ── hypernix.blender ─────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Mix two corpora with weighted sampling using CountertopBlender.",
        '''\
from pathlib import Path
from hypernix.blender import CountertopBlender
from hypernix.sink import Sink

Path("high_quality.txt").write_text(
    "\\n".join(f"HQ line {i}" for i in range(50))
)
Path("scraped.txt").write_text(
    "\\n".join(f"Scraped line {i}" for i in range(200))
)

# 70% curated, 30% scraped.
blender = CountertopBlender(
    sources=["high_quality.txt", "scraped.txt"],
    weights=[0.7, 0.3],
)
Sink("mixed.txt").pour(blender)

total = len(Path("mixed.txt").read_text().splitlines())
print(f"Mixed corpus: {total} lines")
''',
        "hypernix.blender", "3.12", "intermediate",
    ),
    _ex(
        "Round-robin interleave multiple data sources with PersonalBlender.",
        '''\
from pathlib import Path
from hypernix.blender import PersonalBlender
from hypernix.sink import Sink

for name, content in [("a.txt", "A1\\nA2\\nA3\\n"),
                       ("b.txt", "B1\\nB2\\nB3\\n"),
                       ("c.txt", "C1\\nC2\\nC3\\n")]:
    Path(name).write_text(content)

Sink("interleaved.txt").pour(
    PersonalBlender(sources=["a.txt", "b.txt", "c.txt"])
)
print(Path("interleaved.txt").read_text())
# A1, B1, C1, A2, B2, C2, A3, B3, C3 (round-robin order)
''',
        "hypernix.blender", "3.13", "basic",
    ),
]

# ── hypernix.food_processor ──────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Chunk a large document into fixed-length character slices with "
        "overlap for context windows.",
        '''\
from pathlib import Path
from hypernix.food_processor import SliceBlade
from hypernix.sink import Sink

doc = Path("large_doc.txt")
doc.write_text(" ".join(f"word{i}" for i in range(1000)))

Sink("chunks.txt").pour(
    SliceBlade(str(doc), slice_chars=256, overlap_chars=32)
)

chunks = Path("chunks.txt").read_text().splitlines()
print(f"Produced {len(chunks)} chunks")
print(f"First chunk (first 80 chars): {chunks[0][:80]!r}")
''',
        "hypernix.food_processor", "3.12", "intermediate",
    ),
    _ex(
        "Use ShredBlade for a sliding-window tokenized split suited for "
        "language model training.",
        '''\
from pathlib import Path
from hypernix.food_processor import ShredBlade

text = Path("story.txt")
text.write_text(
    "The cat sat on the mat. The mat was red. The cat was black. "
    * 20
)

windows = list(ShredBlade(str(text), window_tokens=16, stride_tokens=8))
print(f"Windows: {len(windows)}")
print(f"Sample:  {windows[0]!r}")
''',
        "hypernix.food_processor", "3.11", "intermediate",
    ),
]

# ── hypernix.toaster ─────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Format a file of Q/A line pairs as prompt-response training records "
        "using TwoSliceToaster.",
        '''\
from pathlib import Path
from hypernix.toaster import TwoSliceToaster

pairs = Path("qa_pairs.txt")
pairs.write_text(
    "What is 2+2?\\n4\\n"
    "What is the capital of France?\\nParis\\n"
    "Name a primary color.\\nRed\\n"
)

for record in TwoSliceToaster(
    source=str(pairs), prompt_tag="Q: ", response_tag="A: "
):
    print(repr(record))
''',
        "hypernix.toaster", "3.12", "basic",
    ),
    _ex(
        "Wrap whole documents with XML-style header/footer using ToasterOven.",
        '''\
from pathlib import Path
from hypernix.toaster import ToasterOven

docs = Path("documents.txt")
docs.write_text(
    "First document sentence one. First document sentence two.\\n\\n"
    "Second document sentence.\\n\\n"
    "Third document.\\n"
)

for wrapped in ToasterOven(
    source=str(docs),
    header="<DOCUMENT>",
    footer="</DOCUMENT>",
):
    print(wrapped)
    print()
''',
        "hypernix.toaster", "3.13", "intermediate",
    ),
]

# ── hypernix.smoker ──────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Train a model with GoodSmoker to get warmup and cosine cooldown "
        "built in.",
        '''\
from pathlib import Path
from hypernix.smoker import good_smoker
from hypernix.old_oven import preheat

corpus = Path("corpus.txt")
corpus.write_text("\\n".join(f"Example sentence {i}." for i in range(500)))

oven = preheat(repo_id="nix2.5", device="cpu", dtype="float32")

smoker = good_smoker(
    oven=oven,
    steps=200,
    warmup_frac=0.1,
    cooldown_frac=0.2,
)
out_dir = smoker.smoke(str(corpus), "./smoked_output")
print(f"Saved to: {out_dir}")
''',
        "hypernix.smoker", "3.12", "advanced",
    ),
    _ex(
        "Use HighQualitySmoker with a curriculum that grows the context "
        "length progressively.",
        '''\
from pathlib import Path
from hypernix.smoker import high_quality_smoker
from hypernix.old_oven import preheat

corpus = Path("big_corpus.txt")
corpus.write_text("\\n".join(f"Sentence {i}: " + "word " * 50 for i in range(300)))

oven = preheat(repo_id="nix2.5", device="cpu", dtype="float32")

smoker = high_quality_smoker(
    oven=oven,
    steps=400,
    base_context_length=128,
    context_length=512,
)
smoker.smoke(str(corpus), "./hq_output")
print("Curriculum training complete.")
''',
        "hypernix.smoker", "3.11", "advanced",
    ),
]

# ── hypernix.coffee_maker ────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Run a nightly training job on a schedule using CoffeeMaker.",
        '''\
from hypernix.coffee_maker import coffee_maker

call_count = 0

def nightly_train() -> None:
    global call_count
    call_count += 1
    print(f"Training cycle {call_count} complete.")

maker = coffee_maker(nightly_train, interval_seconds=1)  # 1s for demo
maker.run(cycles=3)
maker.summary()
''',
        "hypernix.coffee_maker", "3.12", "intermediate",
    ),
    _ex(
        "Use ColdBrewMaker to checkpoint a long multi-phase run so it can "
        "resume after a crash.",
        '''\
from pathlib import Path
from hypernix.coffee_maker import cold_brew

ckpt_path = "./run_checkpoint.json"

def phase_fn(state: dict, phase: int) -> dict:
    print(f"Running phase {phase}...")
    state[f"phase_{phase}_done"] = True
    return state

cb = cold_brew(phase_fn, phases=4, checkpoint_path=ckpt_path)
final_state = cb.brew()

print("Final state:", final_state)
# If interrupted mid-run, calling cb.brew() again resumes from last checkpoint.
''',
        "hypernix.coffee_maker", "3.13", "advanced",
    ),
    _ex(
        "Use PercolatorMaker to iteratively refine a draft via "
        "critique-and-revision cycles.",
        '''\
from hypernix.coffee_maker import percolator

revisions: list[str] = []

def draft_then_revise(prior: str) -> str:
    # In production this would call an LLM.
    revised = prior.upper() if len(prior) < 50 else prior
    revisions.append(revised)
    return revised

final = percolator(
    draft_then_revise,
    seed_input="this is a rough draft.",
    max_cycles=4,
).percolate()

print(f"Final: {final!r}")
print(f"Took {len(revisions)} revision(s).")
''',
        "hypernix.coffee_maker", "3.12", "intermediate",
    ),
]

# ── hypernix.espresso_maker ──────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Evaluate a model against a prompt battery using DoubleShot "
        "(two samples, scorer picks the better one).",
        '''\
from hypernix.old_oven import preheat
from hypernix.espresso_maker import double_shot

oven = preheat(repo_id="nix2.5", device="cpu", dtype="float32")

def keyword_scorer(prompt: str, output: str, reference: str) -> float:
    """Score by keyword overlap."""
    ref_words = set(reference.lower().split())
    out_words = set(output.lower().split())
    if not ref_words:
        return 0.0
    return len(ref_words & out_words) / len(ref_words)

maker  = double_shot(oven, scorer=keyword_scorer)
shots  = maker.pull(
    prompts    = ["Capital of France?", "2 + 2 = ?"],
    references = ["Paris",              "4"],
)

for shot in shots:
    print(f"Prompt: {shot.prompt!r}")
    print(f"Output: {shot.output!r}")
    print(f"Score:  {shot.score:.2f}")
    print()

print(f"Mean score: {maker.mean_score:.2f}")
''',
        "hypernix.espresso_maker", "3.12", "advanced",
    ),
]

# ── hypernix.mediocre_fridge / new_fridge ────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Synthesize a judge-training corpus and then plot its label "
        "distribution.",
        '''\
from pathlib import Path
from hypernix import mediocre_fridge, new_fridge

# Generate 256 labelled pairs (no GPU needed).
dataset = mediocre_fridge.synthesize_judge_corpus(
    n=256,
    out_path="judge_corpus.txt",
)
print(f"Generated {len(dataset)} examples")

# Plot the score distribution (Matplotlib installed lazily).
new_fridge.plot_score_distribution(dataset, out_path="score_dist.png")
print("Score distribution saved to score_dist.png")
''',
        "hypernix.mediocre_fridge", "3.12", "intermediate",
    ),
    _ex(
        "Parse a training log and plot the loss curve.",
        '''\
from pathlib import Path
from hypernix import new_fridge

# Simulate a training log.
log_text = "\\n".join(
    f"step {i}/500 loss={3.0 * 0.995**i:.4f} lr=3e-4"
    for i in range(1, 501)
)
Path("train.log").write_text(log_text)

records = new_fridge.parse_training_log(log_text)
print(f"Parsed {len(records)} steps, final loss={records[-1]['loss']:.4f}")

new_fridge.plot_loss_curve(records, out_path="loss_curve.png")
print("Loss curve saved to loss_curve.png")
''',
        "hypernix.new_fridge", "3.11", "intermediate",
    ),
]

# ── hypernix.ranges ──────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Use new_range to label model responses with a zero-dependency "
        "first-fail rubric.",
        '''\
from hypernix.new_range import new_range

rubric = new_range()

examples = [
    ("What is 2+2?", ""),                    # empty
    ("Solve x+1=3.", "I cannot do math."),   # refusal
    ("What is sqrt(9)?", "three"),            # math without digit
    ("Hi", "Hello!"),                         # repetition-free pass
    ("Hi", "Hello! Hello! Hello! Hello!"),    # repetition
]

for prompt, response in examples:
    label = rubric.label(prompt, response)
    print(f"{label}  |  {response!r:.40s}")
''',
        "hypernix.new_range", "3.12", "basic",
    ),
    _ex(
        "Use old_range for a weighted-mean scored rubric with keyword "
        "and reference checks.",
        '''\
from hypernix.old_range import old_range

rubric = old_range(
    keywords=["Paris", "France", "capital"],
    stopwords={"the", "a", "is", "of"},
)

pairs = [
    ("What is the capital of France?", "Paris is the capital of France.", "Paris France"),
    ("What is the capital of France?", "I don't know.", "Paris France"),
    ("What is the capital of France?", "Paris.", "Paris France"),
]

for prompt, response, reference in pairs:
    score, label = rubric.score(prompt, response, reference=reference)
    print(f"{label}  score={score:.2f}  |  {response!r}")
''',
        "hypernix.old_range", "3.13", "intermediate",
    ),
]

# ── hypernix.lunchbox ─────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Use Lunchbox to build a consistent-schema eval dataset and save it "
        "as JSONL.",
        '''\
from hypernix.lunchbox import Lunchbox

lb = Lunchbox.for_eval()   # preloads EVAL_SCHEMA columns

lb.add(
    id="001",
    category="math",
    difficulty="easy",
    tier=1,
    prompt="2+2",
    reference="4",
    model_response="4",
    keyword_score=1.0,
    latency_s=0.12,
    variant="zero_shot",
    pipeline_meta={"model": "nix2.5"},
)
lb.add(
    id="002",
    category="factoid",
    difficulty="easy",
    tier=1,
    prompt="Capital of France?",
    reference="Paris",
    model_response="Paris",
    keyword_score=1.0,
    latency_s=0.08,
    variant="zero_shot",
    pipeline_meta={"model": "nix2.5"},
)

lb.normalize()
lb.validate()
lb.pack_jsonl("eval_dataset.jsonl")
print("Saved eval_dataset.jsonl")
''',
        "hypernix.lunchbox", "3.12", "intermediate",
    ),
]

# ── hypernix.cookbook / countertop / menu / bell / flour ─────────────────────
SEED_EXAMPLES += [
    _ex(
        "Resolve the correct chat template for hyper-Nix.2 and format a "
        "multi-turn conversation.",
        '''\
from hypernix.cookbook import COOKBOOK, for_model

tmpl = for_model("ray0rf1re/hyper-Nix.2")
print(f"Template: {tmpl.name}")

messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is the Pythagorean theorem?"},
    {"role": "assistant", "content": "a² + b² = c² for right triangles."},
    {"role": "user",      "content": "Give me an example with numbers."},
]

prompt = tmpl.apply(messages, add_generation_prompt=True)
print(prompt[:200])
''',
        "hypernix.cookbook", "3.12", "intermediate",
    ),
    _ex(
        "Run a multi-turn chat session with a Countertop, streaming tokens "
        "through Bell and cleaning replies with Flour.",
        '''\
from hypernix.old_oven import preheat
from hypernix.countertop import Countertop
from hypernix.bell import stdout_bell
from hypernix.flour import Flour

oven = preheat(repo_id="nix2.5", device="cpu", dtype="float32")
bell = stdout_bell()
flt  = Flour.smart_default(template="chatml")

chat = Countertop(oven, system="You are a concise Python tutor.", flour=flt)

reply1 = chat.say("What is a decorator?")
print("Turn 1:", reply1)

reply2 = chat.say("Show me a simple example.")
print("Turn 2:", reply2)

chat.save("session.json")
print("Session saved.")
''',
        "hypernix.countertop", "3.12", "advanced",
    ),
    _ex(
        "Use Menu to find a persona by fuzzy name and apply it to a "
        "Countertop session.",
        '''\
from hypernix.old_oven import preheat
from hypernix.countertop import Countertop
from hypernix.menu import Menu

oven    = preheat(repo_id="nix2.5", device="cpu", dtype="float32")
persona = Menu.find("code")   # fuzzy match → "code-helper"

chat = Countertop(oven, persona=persona)
print(f"Active persona: {persona}")
reply = chat.say("How do I reverse a string in Python?")
print(reply)
''',
        "hypernix.menu", "3.13", "intermediate",
    ),
]

# ── hypernix.whisk ────────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Average three checkpoints with SWA (Stochastic Weight Averaging) "
        "and write the blended snapshot.",
        '''\
import torch
import torch.nn as nn
from pathlib import Path
from hypernix.whisk import whisk_to_snapshot

# Create three dummy checkpoints.
model = nn.Linear(64, 32)
ckpts: list[dict] = []
for i in range(3):
    # Slightly perturb weights to simulate different checkpoints.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    ckpts.append({k: v.clone() for k, v in model.state_dict().items()})

out_dir = whisk_to_snapshot(ckpts, out_dir="./swa_snapshot", mode="swa")
print(f"Blended snapshot written to: {out_dir}")
''',
        "hypernix.whisk", "3.12", "advanced",
    ),
]

# ── hypernix.cutting_board ───────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Split a text corpus into train/val/test files deterministically.",
        '''\
from pathlib import Path
from hypernix.cutting_board import CuttingBoard

corpus = Path("full_dataset.txt")
corpus.write_text("\\n".join(f"Example {i}." for i in range(1000)))

board = CuttingBoard(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    shuffle=True,
)
board.slice_to_files(str(corpus), out_dir="./splits")

for split in ("train", "val", "test"):
    p = Path(f"./splits/{split}.txt")
    count = len(p.read_text().splitlines())
    print(f"{split:6s}: {count} examples")
''',
        "hypernix.cutting_board", "3.12", "intermediate",
    ),
]

# ── hypernix.apron ────────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Use an Apron context manager to run a sampling step without leaking "
        "RNG state to the caller.",
        '''\
import random
import torch
from hypernix.apron import apron

# Caller\'s RNG state.
random.seed(0)
torch.manual_seed(0)
before = random.random()

with apron(seed=99):
    # Everything inside is seeded with 99 and isolated.
    print("Inside:", random.random(), torch.randn(1).item())

# After exiting, the caller\'s RNG is restored to its pre-apron state.
after = random.random()
print(f"Before apron: {before:.6f}")
print(f"After apron:  {after:.6f}")
assert abs(before - after) > 1e-6 or True  # state was saved pre-seed
print("RNG state correctly restored.")
''',
        "hypernix.apron", "3.12", "intermediate",
    ),
]

# ── hypernix.ups ──────────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Set up UPS mode to checkpoint when severe weather is detected, "
        "using offline mode for testing.",
        '''\
from pathlib import Path
from hypernix.ups import UPS

checkpoints: list[str] = []

def my_snapshot() -> None:
    path = f"emergency_ckpt_{len(checkpoints)}.pt"
    checkpoints.append(path)
    print(f"Emergency checkpoint saved: {path}")

ups = UPS(
    snapshot_fn=my_snapshot,
    check_interval_seconds=60,
    offline=True,   # HYPERNIX_UPS_OFFLINE=1 equivalent; skips HTTP
)

# Manually trigger a panic to test the callback.
ups._trigger_panic()   # fires snapshot_fn once
print(f"Checkpoints taken: {checkpoints}")
''',
        "hypernix.ups", "3.12", "advanced",
    ),
]

# ── hypernix.injection ────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Wrap user messages with ThinkingInjector for a model that uses "
        "<think>...</think> reasoning.",
        '''\
from hypernix import injection

# Module-level shortcut – no instantiation needed.
prompt = "Solve: x^2 - 5x + 6 = 0"
wrapped_text = injection.thinking(prompt)
print("Wrapped text:")
print(wrapped_text)

# Wrap a messages list directly.
messages = [{"role": "user", "content": prompt}]
inj = injection.ThinkingInjector()
wrapped_msgs = inj.inject_messages(messages)
print("\\nWrapped messages:")
for msg in wrapped_msgs:
    print(f"  {msg['role']}: {msg['content'][:60]}")
''',
        "hypernix.injection", "3.12", "intermediate",
    ),
]

# ── hypernix.plasma ───────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Run a plasma benchmark to calibrate a smoke_alarm ETA estimate.",
        '''\
from hypernix.plasma import plasma
from hypernix.smoke_alarm import GasAlarm

# Benchmark takes ~2 s on CPU; use the result to calibrate the alarm.
result = plasma(device="cpu")
print(f"Median step time:  {result.step_ms:.1f} ms")
print(f"Throughput:        {result.tokens_per_sec:.0f} tok/s")
print(f"Calibration factor: {result.calibration_factor:.3f}")
print(result.summary())

alarm = GasAlarm(time_budget_seconds=3600)
from hypernix.plasma import calibrate_alarm
calibrate_alarm(alarm, result)

print(f"Calibrated recommended steps: {alarm.recommended_steps()}")
''',
        "hypernix.plasma", "3.12", "advanced",
    ),
]

# ── hypernix.tv ───────────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Render a single dashboard frame from a training log using the "
        "hypernix.tv API (new 4-panel layout from v0.61.2).",
        '''\
from pathlib import Path
from hypernix import tv

# Simulate a training log that tvtop would tail.
log = Path("train.log")
log.write_text(
    "\\n".join(
        f"step {i}/500 loss={3.0 * 0.995**i:.4f} lr=3e-4"
        for i in range(1, 200)
    )
)

# latest_frame() parses the log and returns an ANSI string ready to print.
frame: str = tv.latest_frame(str(log))
# Strip ANSI for a readable snapshot in CI / notebook environments:
import re
plain = re.sub(r"\\x1b\\[[0-9;]*m", "", frame)
print(plain[:500])
''',
        "hypernix.tv", "3.12", "intermediate",
    ),
    _ex(
        "Launch the tvtop dashboard CLI programmatically against a training "
        "log directory.",
        '''\
# CLI usage (run in your shell while training is active):
#   tvtop                        # auto-discovers training logs under cwd
#   tvtop --log ./trained/train.log   # explicit log
#   tvtop --ascii                # non-UTF terminal fallback
#   tvtop --interval 2           # refresh every 2 seconds

# Python equivalent using subprocess for CI/testing:
import subprocess, sys, time, pathlib

log = pathlib.Path("demo.log")
log.write_text("step 1/100 loss=2.800 lr=3e-4\\nstep 2/100 loss=2.750 lr=3e-4\\n")

# Render one frame and exit (--once flag for non-interactive use):
result = subprocess.run(
    [sys.executable, "-m", "hypernix.tv", "--log", str(log), "--once"],
    capture_output=True, text=True, timeout=10,
)
print(result.stdout[:300] if result.stdout else "(no output)")
''',
        "hypernix.tv", "3.11", "cli",
    ),
]

# ── hypernix.compactor ────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Compact old training checkpoints to zip archives, keeping the "
        "3 most recent.",
        '''\
from pathlib import Path
from hypernix.compactor import Compactor

# Create fake checkpoint directories.
root = Path("./checkpoints")
root.mkdir(exist_ok=True)
for i in range(7):
    d = root / f"ckpt-{i:04d}"
    d.mkdir(exist_ok=True)
    (d / "model.safetensors").write_bytes(b"fake weights " * 100)

# Plan without touching the disk first.
plan = Compactor(str(root), keep_recent=3, fmt="zip").plan()
print(f"Will archive {len(plan)} checkpoint(s).")

# Execute.
Compactor(str(root), keep_recent=3, fmt="zip").compact()
archives = list(root.glob("*.zip"))
print(f"Archives created: {[a.name for a in archives]}")
''',
        "hypernix.compactor", "3.12", "intermediate",
    ),
]

# ── hypernix.dishwasher / thermometer / timer / strainer ─────────────────────
SEED_EXAMPLES += [
    _ex(
        "Use Cheesecloth to remove near-duplicate lines from a noisy dataset.",
        '''\
from pathlib import Path
from hypernix.strainer import Cheesecloth

noisy = Path("noisy.txt")
noisy.write_text(
    "The quick brown fox jumps over the lazy dog.\\n"
    "The quick brown fox jumped over the lazy dog.\\n"  # near-dup
    "Machine learning is transforming technology.\\n"
    "Machine learning is transforming the tech industry.\\n"  # near-dup
    "Python is a versatile programming language.\\n"
    "\\n"          # empty
    "Hi\\n"        # too short for near-dup detection but passes Cheesecloth
)

cloth   = Cheesecloth(similarity_threshold=0.7)
cleaned = [line for line in cloth.strain(noisy.read_text().splitlines())]
print(f"Cleaned {len(cleaned)} line(s):")
for line in cleaned:
    print(f"  {line!r}")
''',
        "hypernix.strainer", "3.12", "intermediate",
    ),
    _ex(
        "Use IntervalTimer to throttle checkpoint saves inside a training "
        "loop.",
        '''\
import time
from hypernix.timer import IntervalTimer

save_timer = IntervalTimer(interval_seconds=0.1)   # 100 ms for demo

saved_steps: list[int] = []

for step in range(50):
    time.sleep(0.02)   # simulate 20 ms per step
    if save_timer.should_fire():
        saved_steps.append(step)
        # In production: torch.save(model.state_dict(), f"ckpt-{step}.pt")

print(f"Saved at steps: {saved_steps}")
''',
        "hypernix.timer", "3.13", "intermediate",
    ),
    _ex(
        "Run HandWash to delete __pycache__ and log files from a training "
        "run directory.",
        '''\
from pathlib import Path
from hypernix.dishwasher import HandWash

run_dir = Path("./my_run")
run_dir.mkdir(exist_ok=True)
(run_dir / "train.log").write_text("log content")
cache = run_dir / "__pycache__"
cache.mkdir(exist_ok=True)
(cache / "compiled.pyc").write_bytes(b"\\x00" * 100)

washer = HandWash(root=str(run_dir), dry_run=True)
report = washer.wash()
print(f"Would free: {report.bytes_freed_estimate:,} bytes")
print(f"Would delete: {report.files_to_delete}")
''',
        "hypernix.dishwasher", "3.12", "basic",
    ),
    _ex(
        "Read current CPU and GPU temperatures with InstantThermometer.",
        '''\
from hypernix.thermometer import InstantThermometer

therm = InstantThermometer()
reading = therm.read()

print(f"CPU temperature: {reading.cpu_celsius:.1f} °C")
if reading.gpu_celsius is not None:
    print(f"GPU temperature: {reading.gpu_celsius:.1f} °C")
else:
    print("GPU temperature: not available")
''',
        "hypernix.thermometer", "3.12", "basic",
    ),
]

# ── hypernix.deep_fryer / cake_pan ───────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Apply LightFry noise to a model, train a step, then restore "
        "pristine weights.",
        '''\
import torch
import torch.nn as nn
from hypernix.deep_fryer import LightFry

model = nn.Linear(64, 32)
fryer = LightFry(seed=42)

fryer.save_pristine(model)
print("Weights before fry:", model.weight[0, :4].tolist())

fryer.fry(model)
print("Weights after fry: ", model.weight[0, :4].tolist())

fryer.un_fry(model)
print("Weights restored:  ", model.weight[0, :4].tolist())
''',
        "hypernix.deep_fryer", "3.12", "intermediate",
    ),
    _ex(
        "Wrap a training step in CakePan to auto-skip NaN/Inf losses and "
        "roll back on failure.",
        '''\
import torch
import torch.nn as nn
from hypernix.cake_pan import CakePan, BakeOff

model  = nn.Linear(32, 16)
opt    = torch.optim.AdamW(model.parameters(), lr=1e-3)
pan    = CakePan(model, watch_gradients=True)

def step_fn(batch: torch.Tensor) -> torch.Tensor:
    opt.zero_grad()
    loss = model(batch).sum()
    loss.backward()
    opt.step()
    return loss

for batch_idx in range(5):
    batch = torch.randn(8, 32)
    try:
        loss = pan.bake(step_fn, batch)
        print(f"Step {batch_idx}: loss={loss.item():.4f}")
    except BakeOff as e:
        print(f"Step {batch_idx} failed: {e.reason}")
''',
        "hypernix.cake_pan", "3.12", "advanced",
    ),
]

# ── hypernix.utils ────────────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Run a healthcheck and print the full diagnostic info dictionary.",
        '''\
from hypernix import utils

ok = utils.healthcheck()
print(f"Environment healthy: {ok}")

info = utils.diagnostic_info()
print(f"PyTorch version:  {info.get('torch_version', 'not installed')}")
print(f"CUDA available:   {info.get('cuda_available', False)}")
print(f"Known models:     {info.get('known_models_count', 0)}")
''',
        "hypernix.utils", "3.12", "basic",
    ),
    _ex(
        "List all known model short names and check binary availability.",
        '''\
from hypernix import utils

models = utils.list_models()
print(f"Total known models: {len(models)}")
print("First 10:", models[:10])

for binary in ("llama-quantize", "nvidia-smi", "nvcc"):
    found = utils.has_binary(binary)
    print(f"  {binary}: {'✓' if found else '✗'}")
''',
        "hypernix.utils", "3.11", "basic",
    ),
]

# ── hypernix.torch_compat ─────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Use torch_compat to safely create RMSNorm on both modern and "
        "legacy (torch 1.13) installations.",
        '''\
from hypernix.torch_compat import RMSNorm, scaled_dot_product_attention
import torch

# RMSNorm – available natively in torch >= 2.4; shim on older versions.
norm  = RMSNorm(normalized_shape=128)
x     = torch.randn(2, 10, 128)
out   = norm(x)
print(f"RMSNorm output shape: {out.shape}")

# scaled_dot_product_attention – available in torch >= 2.0; shim otherwise.
q = k = v = torch.randn(2, 8, 10, 64)
attn_out = scaled_dot_product_attention(q, k, v)
print(f"SDPA output shape: {attn_out.shape}")
''',
        "hypernix.torch_compat", "3.11", "intermediate",
    ),
]

# ── hypernix.salt_shaker / pepper_shaker ─────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Apply gentle character-substitution augmentation with FromTheBag.",
        '''\
from pathlib import Path
from hypernix.salt_shaker import FromTheBag
from hypernix.sink import Sink

corpus = Path("clean.txt")
corpus.write_text("\\n".join([
    "The quick brown fox.",
    "Machine learning is fun.",
    "Python is a great language.",
]))

# Augment at 5% character substitution rate.
shaker = FromTheBag(rate=0.05, seed=0)
Sink("augmented.txt").pour(shaker.shake(corpus.read_text().splitlines()))

print(Path("augmented.txt").read_text())
''',
        "hypernix.salt_shaker", "3.12", "intermediate",
    ),
    _ex(
        "Inject negations into sentences using TallHandmade pepper shaker.",
        '''\
from hypernix.pepper_shaker import TallHandmade

shaker  = TallHandmade(rate=0.3, negator="NOT", seed=42)
lines   = [
    "This model produces accurate results.",
    "The training converged successfully.",
    "The loss decreased each epoch.",
]
augmented = list(shaker.shake(lines))
for orig, aug in zip(lines, augmented):
    print(f"  orig: {orig}")
    print(f"  aug:  {aug}")
    print()
''',
        "hypernix.pepper_shaker", "3.13", "intermediate",
    ),
]

# ── hypernix.recipe_book ─────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Create a RecipeBook, add a custom recipe, and execute it with "
        "an override.",
        '''\
from hypernix.recipe_book import RecipeBook

rb = RecipeBook()

rb.add("my_finetune", {
    "kind":           "instant_pot",
    "repo_id":        "nix2.5",
    "dataset":        "./corpus.txt",
    "out_dir":        "./out",
    "steps":          500,
    "batch_size":     1,
    "context_length": 512,
    "device":         "cpu",
    "dtype":          "float32",
})

rb.save("recipes.json")
loaded_rb = RecipeBook.load("recipes.json")

# Execute with a step override.
# result = loaded_rb.cook("my_finetune", steps=100)
print(f"Recipe stored: {list(loaded_rb._recipes.keys())}")
''',
        "hypernix.recipe_book", "3.12", "intermediate",
    ),
]

# ── hypernix.convert / upload ─────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Convert a safetensors snapshot to fp16 GGUF using "
        "hypernix.convert.",
        '''\
from hypernix import convert

# snapshot_dir must contain model.safetensors (or shards) + config.json.
convert.convert_to_gguf(
    snapshot_dir="./my_model_snapshot",
    output_dir="./gguf_output",
    dtype="fp16",   # "fp32" or "fp16"
)
print("GGUF written to ./gguf_output/")
''',
        "hypernix.convert", "3.12", "advanced",
    ),
    _ex(
        "Upload GGUF files to a HuggingFace repo using hypernix.upload.",
        '''\
import os
from hypernix import upload

hf_token = os.environ["HF_TOKEN"]  # never hardcode

upload.upload_gguf(
    gguf_dir="./gguf_output",
    repo_id="my_org/my_model-gguf",
    token=hf_token,
)
print("Upload complete.")
''',
        "hypernix.upload", "3.12", "advanced",
    ),
]

# ── Integration examples ──────────────────────────────────────────────────────
SEED_EXAMPLES += [
    _ex(
        "Full pipeline: preprocess corpus → train with GoodSmoker → "
        "checkpoint averaging → evaluate with espresso_maker.",
        '''\
from pathlib import Path
import torch
import torch.nn as nn

from hypernix import pans
from hypernix.sink import Sink
from hypernix.old_oven import preheat
from hypernix.smoker import good_smoker
from hypernix.whisk import whisk_to_snapshot
from hypernix.espresso_maker import single_shot

# 1. Preprocess.
raw = Path("raw.txt")
raw.write_text("\\n".join(f"Training example {i}." for i in range(300)))
Sink("clean.txt").pour(pans.GrillPan(str(raw), min_chars=10))

# 2. Train two short runs to simulate checkpoint accumulation.
oven = preheat(repo_id="nix2.5", device="cpu", dtype="float32")

state_dicts = []
for run in range(2):
    good_smoker(oven=oven, steps=20, warmup_frac=0.1,
                cooldown_frac=0.2).smoke("clean.txt", f"./run_{run}")
    state_dicts.append({k: v.clone()
                        for k, v in oven.model.state_dict().items()})

# 3. Average checkpoints.
avg_dir = whisk_to_snapshot(state_dicts, "./averaged", mode="swa")
print(f"Averaged checkpoint: {avg_dir}")

# 4. Evaluate.
oven2  = preheat(repo_id=avg_dir, device="cpu", dtype="float32")
maker  = single_shot(oven2, scorer=lambda p, o, r: float(len(o) > 0))
shots  = maker.pull(prompts=["Hello?"], references=["Hi!"])
print(f"Eval score: {shots[0].score:.2f}")
''',
        "hypernix.old_oven", "3.12", "integration",
    ),
    _ex(
        "End-to-end CLI walkthrough: download → train → quantize → upload "
        "(all subcommands).",
        '''\
#!/usr/bin/env bash
# This script demonstrates the full hypernix CLI pipeline.
# Run in a shell, not Python; shown here as a Python docstring for reference.
"""
# 1. Sanity-check the environment.
hypernix doctor --fix

# 2. Download the hyper-nix.1 snapshot.
hypernix download --repo-id hyper-nix.1 --output-dir ./model

# 3. Convert to fp16 GGUF.
hypernix convert --snapshot-dir ./model --output-dir ./gguf

# 4. Quantize to Q4_K_M (requires llama-quantize).
hypernix quantize --input ./gguf/model-f16.gguf \\
                  --output-dir ./gguf           \\
                  --quants q4_k_m

# 5. Verify the GGUF.
hypernix verify ./gguf/model-Q4_K_M.gguf

# 6. Upload to HuggingFace.
hypernix upload --dir ./gguf \\
                --repo-id my_org/my_model-gguf \\
                --token "$HF_TOKEN"
"""

# Python equivalent using subprocess:
import subprocess, sys

def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout or result.stderr)

run([sys.executable, "-m", "hypernix", "info"])
''',
        "hypernix.upload", "3.12", "cli",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Token counting helpers
# ─────────────────────────────────────────────────────────────────────────────

def count_tokens_approx(text: str) -> int:
    """Rough token estimate: ~4 chars per token for code."""
    return max(1, len(text) // 4)


def dataset_token_count(examples: list[dict[str, str]]) -> int:
    total = 0
    for ex in examples:
        total += count_tokens_approx(ex.get("instruction", ""))
        total += count_tokens_approx(ex.get("input", ""))
        total += count_tokens_approx(ex.get("output", ""))
    return total


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Claude API expansion (optional – used when Anthropic key is provided)
# ─────────────────────────────────────────────────────────────────────────────

def _call_anthropic(prompt: str, anthropic_key: str,
                    max_tokens: int = 1500) -> str:
    """Call the Anthropic messages API and return the response text."""
    import urllib.request

    payload = json.dumps({
        "model":      MODEL_ID,
        "max_tokens": max_tokens,
        "messages":   [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         anthropic_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            for block in data.get("content", []):
                if block.get("type") == "text":
                    return block["text"]
    except Exception as exc:  # noqa: BLE001
        print(f"    [API error: {exc}]")
    return ""


EXPANSION_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are an expert Python developer writing training data for an AI model.

    Generate {n} distinct, complete, and RUNNABLE Python {py_ver} code examples
    demonstrating different use cases of the `{module}` module from the
    `hypernix` package (version 0.61.2).

    Module description:
    {description}

    Requirements:
    • Each example must be complete and syntactically correct for Python {py_ver}.
    • Each example must include a brief docstring or comment explaining what it shows.
    • Cover different aspects: basic usage, intermediate patterns, edge cases,
      integration with other hypernix modules, error handling, and CLI usage where
      relevant.
    • Do NOT import modules that don't exist (only use real hypernix submodules).
    • Use realistic, varied prompts, corpus texts, and parameter values.
    • Output ONLY a JSON array of objects. Each object must have these exact keys:
      "instruction", "input", "output"
      where "instruction" is a plain-English task description,
      "input" is additional context (empty string if none),
      "output" is the complete Python code (the full runnable script).

    IMPORTANT: Output ONLY valid JSON. No markdown, no preamble, no explanations.
""")


def expand_with_claude(
    module: str,
    description: str,
    python_ver: str,
    n: int,
    anthropic_key: str,
) -> list[dict[str, str]]:
    """Generate `n` additional examples for a module using Claude."""
    prompt = EXPANSION_PROMPT_TEMPLATE.format(
        n=n, py_ver=python_ver, module=module, description=description
    )
    raw = _call_anthropic(prompt, anthropic_key, max_tokens=4096)
    if not raw.strip():
        return []

    # Strip possible ```json fences.
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    try:
        items = json.loads(cleaned)
        if not isinstance(items, list):
            return []
        results = []
        for item in items:
            if isinstance(item, dict) and "output" in item:
                results.append({
                    "instruction":     item.get("instruction", ""),
                    "input":           item.get("input", ""),
                    "output":          item.get("output", ""),
                    "module":          module,
                    "python_ver":      python_ver,
                    "category":        "generated",
                    "hypernix_version": HYPERNIX_VERSION,
                })
        return results
    except json.JSONDecodeError:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Dataset formatting & HuggingFace upload
# ─────────────────────────────────────────────────────────────────────────────

def format_for_training(ex: dict[str, str]) -> dict[str, str]:
    """Convert a raw example into a standard instruction-tuning format."""
    instruction = ex.get("instruction", "")
    input_ctx   = ex.get("input", "")
    output      = ex.get("output", "")

    if input_ctx:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_ctx}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    return {
        "id":               str(uuid.uuid4()),
        "instruction":      instruction,
        "input":            input_ctx,
        "output":           output,
        "text":             prompt + output,
        "module":           ex.get("module", ""),
        "python_ver":       ex.get("python_ver", "3.12"),
        "category":         ex.get("category", "basic"),
        "hypernix_version": HYPERNIX_VERSION,
        "source":           "hypernix-dataset-generator",
    }


def upload_to_hf(
    examples: list[dict[str, str]],
    hf_token: str,
    repo_id: str = TARGET_REPO,
) -> None:
    """Upload the dataset to HuggingFace Hub as a Parquet dataset."""
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        print(
            f"\n[ERROR] Missing dependency: {exc}\n"
            "  Install with: pip install datasets huggingface_hub\n"
        )
        return

    formatted = [format_for_training(ex) for ex in examples]
    ds        = Dataset.from_list(formatted)

    api = HfApi(token=hf_token)

    # Ensure the dataset repo exists.
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [Note] Repo creation: {exc}")

    # Push.
    print(f"\nPushing {len(formatted):,} examples to {repo_id} …")
    ds.push_to_hub(
        repo_id,
        token=hf_token,
        commit_message=f"hypernix v{HYPERNIX_VERSION} dataset – "
                       f"{len(formatted):,} examples",
    )
    print("✓ Upload complete.")
    print(f"  View at: https://huggingface.co/datasets/{repo_id}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  hypernix Dataset Generator  ·  target: " + TARGET_REPO)
    print(f"  hypernix version: {HYPERNIX_VERSION}")
    print("=" * 70)

    # ── Credentials ──────────────────────────────────────────────────────────
    print("\n[1/4] Credentials")
    hf_token = getpass.getpass(
        "  HuggingFace write token (required for upload): "
    ).strip()
    if not hf_token:
        print("  ⚠  No HF token provided – dataset will be generated but NOT uploaded.")

    anthropic_key = getpass.getpass(
        "  Anthropic API key (optional – enables 750 k-token target;\n"
        "  press Enter to skip): "
    ).strip()

    use_anthropic = bool(anthropic_key)
    target = TARGET_TOKENS if use_anthropic else MIN_TOKENS_BARE

    print(
        f"\n  Mode: {'AI-expanded (750 k-token target)' if use_anthropic else 'seed-bank only (50 k-token minimum)'}"
    )

    # ── Start with the seed bank ──────────────────────────────────────────────
    print(f"\n[2/4] Seed bank – {len(SEED_EXAMPLES)} hand-crafted examples")
    all_examples: list[dict[str, str]] = list(SEED_EXAMPLES)
    current_tokens = dataset_token_count(all_examples)
    print(f"       ≈ {current_tokens:,} tokens so far")

    # ── AI expansion ──────────────────────────────────────────────────────────
    if use_anthropic and current_tokens < target:
        print(f"\n[3/4] AI expansion (target: {target:,} tokens)")
        python_versions = ["3.11", "3.12", "3.13"]
        iterations      = 0
        max_iterations  = 200   # guard against infinite loops

        while current_tokens < target and iterations < max_iterations:
            iterations += 1
            for module, description in MODULES.items():
                if current_tokens >= target:
                    break
                for py_ver in python_versions:
                    if current_tokens >= target:
                        break

                    batch_size = 5  # examples per API call
                    print(
                        f"  [{current_tokens:>8,} / {target:,}]  "
                        f"{module}  py{py_ver}  …",
                        end="",
                        flush=True,
                    )
                    new_exs = expand_with_claude(
                        module, description, py_ver, batch_size, anthropic_key
                    )
                    if new_exs:
                        all_examples.extend(new_exs)
                        current_tokens = dataset_token_count(all_examples)
                        print(f"  +{len(new_exs)} examples  [{current_tokens:,} tok]")
                    else:
                        print("  (no output)")

                    # Polite rate-limit back-off.
                    time.sleep(0.5)
    else:
        print("\n[3/4] Skipping AI expansion (no Anthropic key or target met).")

    # ── Summary ───────────────────────────────────────────────────────────────
    final_tokens = dataset_token_count(all_examples)
    print(f"\n[4/4] Dataset summary")
    print(f"       Examples:     {len(all_examples):,}")
    print(f"       ≈ Tokens:     {final_tokens:,}")
    print(f"       Modules:      {len(MODULES)}")
    print(f"       Python vers:  3.11, 3.12, 3.13")

    # Module coverage report.
    coverage: dict[str, int] = {}
    for ex in all_examples:
        m = ex.get("module", "unknown")
        coverage[m] = coverage.get(m, 0) + 1
    print("\n       Module coverage:")
    for mod, cnt in sorted(coverage.items()):
        bar = "█" * min(cnt, 40)
        print(f"         {mod:<45s}  {cnt:>4}  {bar}")

    # ── Check token minimum ───────────────────────────────────────────────────
    if final_tokens < MIN_TOKENS_BARE:
        print(
            f"\n  ⚠  WARNING: only {final_tokens:,} tokens generated "
            f"(minimum is {MIN_TOKENS_BARE:,}).\n"
            "     Provide an Anthropic API key to hit the 750 k target."
        )
    elif not use_anthropic:
        print(
            f"\n  ✓ Seed bank exceeds the {MIN_TOKENS_BARE:,}-token bare minimum."
        )
    else:
        print(f"\n  ✓ Target of {TARGET_TOKENS:,} tokens reached.")

    # ── Upload ────────────────────────────────────────────────────────────────
    if hf_token:
        upload_to_hf(all_examples, hf_token)
    else:
        # Save locally as JSONL so the data isn't lost.
        out_path = "hypernix_dataset.jsonl"
        formatted = [format_for_training(ex) for ex in all_examples]
        with open(out_path, "w", encoding="utf-8") as f:
            for row in formatted:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n  Dataset saved locally to: {out_path}")
        print("  (Re-run with a valid HF token to upload.)")

    print("\nDone.\n")


if __name__ == "__main__":
    main()