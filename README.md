# SDXL serverless worker (RunPod)

One image that can serve **any SDXL-family checkpoint** on a RunPod Serverless
endpoint: diffusers repos, single-file `.safetensors` checkpoints, external VAEs
and LoRAs. Models live on a RunPod **network volume** and are downloaded on
first use (cache-on-first-use), so a cold start never re-downloads weights.

This replaces the previous setup, which reused the stock
`runpod/ai-api-sdxl:2.1.1` image and injected a checkpoint with a base64
`dockerStartCmd` that re-downloaded ~6.5 GB on every cold start and could not
load LoRA at all.

## Layout

    handler.py          RunPod handler (input/output contract, pipeline cache)
    model_store.py      network-volume model store: classify -> fetch -> registry.json
    pipeline_utils.py   torch-free helpers: input parsing, sampler table, clamps
    tests/              40 unit tests, no GPU and no network required
    Dockerfile          cuda runtime base + python 3.11 + pinned wheels
    requirements.txt    pinned dependencies (verified against PyPI)
    .github/workflows/build.yml   builds the image and pushes it to GHCR

## Request / response

```json
{"input": {
  "model": "org/repo",
  "prompt": "a red apple on a wooden table",
  "negative_prompt": "blurry",
  "width": 1024, "height": 1024,
  "steps": 30, "guidance": 6.5, "seed": 1234,
  "sampler": "auto",
  "clip_skip": 2,
  "vae": null,
  "loras": [{"path": "org/lora-repo::lora.safetensors", "weight": 0.8}]
}}
```

`model` / `vae` accept:

| form | meaning |
|---|---|
| `org/repo` | diffusers repo (snapshot) |
| `org/repo::file.safetensors` | single weight file from a repo |
| `org/repo/file.safetensors` | same, short form |
| `https://…/file.safetensors` | direct download |
| `/runpod-volume/models/…` | already on the volume |

Notes:

* `model` may be omitted — the endpoint's `MODEL_ID` env var is then used.
* `num_inference_steps` / `guidance_scale` are accepted as aliases (that is what
  the bot already sends), and the legacy `refiner_inference_steps` /
  `high_noise_frac` keys are ignored instead of failing the job.
* `sampler`: `auto` (keep the checkpoint's own scheduler) or one of `euler`,
  `euler_a`, `dpmpp_2m`, `dpmpp_2m_karras`, `dpmpp_2m_sde`, `dpmpp_sde`, `ddim`,
  `pndm`, `lms`, `heun`, `unipc`.
* `clip_skip` (default 2 = diffusers' own behaviour) re-encodes the prompt with
  `hidden_states[-clip_skip]`.
* Optional `quality_prefix` / `appearance` / `style_positive` fragments are
  prepended in that order when present; without them the prompt is used verbatim.

Response:

```json
{"image": "<base64 PNG>",
 "meta": {"model": "...", "model_type": "hf_repo|hf_single_file|url_file|local_file",
          "model_cached": true, "model_download_seconds": 0.0,
          "pipeline_load_seconds": 8.4, "pipeline_reused": false,
          "gpu_name": "NVIDIA L4", "gpu_total_memory_gb": 22.5,
          "generation_seconds": 3.1, "request_seconds": 12.0,
          "steps": 30, "size": [1024, 1024], "guidance": 6.5,
          "sampler": "auto", "clip_skip": 2, "seed": 1234,
          "loras": [], "image_bytes": 1234567, "volume_available": true}}
```

## Environment variables (endpoint)

| name | default | meaning |
|---|---|---|
| `MODEL_ID` | *(empty)* | default checkpoint |
| `LORA_ID` / `LORA_FILE` / `LORA_SCALE` | *(empty)* | default LoRA when the request does not list any |
| `DEFAULT_STEPS` / `DEFAULT_SIZE` / `DEFAULT_GUIDANCE` | 30 / 1024 / 6.5 | request defaults |
| `MAX_CACHED_PIPELINES` | 2 | how many loaded pipelines stay in GPU memory |
| `ALLOW_DOWNLOAD` | 1 | set `0` to forbid fetching new models |
| `HF_TOKEN` | *(unset)* | only for gated repos |
| `MODEL_ROOT` | `/runpod-volume/models` | store root (volume mount) |

## Adding a model

1. Send one request with the new `model` reference (or set `MODEL_ID`).
2. The worker downloads it into `/runpod-volume/models/checkpoints/` and records
   it in `/runpod-volume/models/registry.json`; the request pays the download.
3. Every later request — including on a fresh worker — loads it from the volume,
   so `meta.model_cached` is `true` and `model_download_seconds` is `0`.

To pre-populate the volume without a GPU request, use RunPod's S3-compatible
API (`docs.runpod.io/storage/s3-api`) and upload into `checkpoints/`, `loras/`
or `vae/`.

## Build

    gh workflow run build-worker-image        # or push to main
    # image: ghcr.io/<owner>/sdxl-serverless-worker:latest

The image is ~5.5 GB, so it is built on GitHub runners, never on the VPS.

## Local checks

    /root/nsfw-bot/bot/venv/bin/python -m pytest -q tests
    python3 -m py_compile handler.py model_store.py pipeline_utils.py

`runpod.serverless.start` also supports a local dry run of the handler logic:

    python3 handler.py --test_input test_input.json
