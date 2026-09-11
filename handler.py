"""RunPod Serverless handler: one worker image, any SDXL-family checkpoint.

Why this exists (kanban t_970eafc9): the previous setup reused
``runpod/ai-api-sdxl:2.1.1`` and smuggled a checkpoint into it with a base64
start command that re-downloaded ~6.5 GB on every cold start and could not load
LoRA. This image owns the pipeline instead:

  * any SDXL-family checkpoint: a diffusers repo, a single-file
    ``.safetensors`` (``repo::file``), a direct URL, or a path on the volume;
  * models live on a network volume and are fetched on first use
    (cache-on-first-use), never re-downloaded;
  * LoRA support, selectable sampler, ``clip_skip``, optional external VAE;
  * the response carries ``meta`` with the GPU name and the real timings.

Request (``input``)::

    {"model": "org/repo" | "org/repo::file.safetensors" | "https://.../x.safetensors"
              | "/runpod-volume/models/checkpoints/foo.safetensors",
     "prompt": "...", "negative_prompt": "...",
     "width": 1024, "height": 1024, "steps": 30, "guidance": 6.5,
     "seed": -1, "sampler": "auto|euler|euler_a|dpmpp_2m|dpmpp_2m_karras|"
                            "dpmpp_2m_sde|dpmpp_sde|ddim|pndm|lms|heun|unipc",
     "clip_skip": 2, "vae": "<same reference forms as model>",
     "loras": [{"path": "...", "weight": 0.8}]}

Response::

    {"image": "<base64 PNG>", "meta": {...}}

``num_inference_steps``/``guidance_scale`` (what bot/generation.py already
sends) and the legacy ``refiner_inference_steps``/``high_noise_frac`` keys are
still accepted.
"""
import base64
import importlib
import io
import logging
import os
import random
import time
import types

import runpod
import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

from model_store import HF_CACHE_ROOT, ensure, host_cache_repos, resolve_root, volume_available
from pipeline_utils import (
    DEFAULT_GUIDANCE,
    DEFAULT_MODEL_ID,
    DEFAULT_NEGATIVE,
    DEFAULT_SAMPLER,
    DEFAULT_SIZE,
    DEFAULT_STEPS,
    clamp_float,
    clamp_int,
    parse_job_input,
    sampler_spec,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sdxl-worker")

# ---- configuration (endpoint env vars) ------------------------------------ #
MODEL_ID = (os.environ.get("MODEL_ID") or DEFAULT_MODEL_ID).strip()
LORA_ID = (os.environ.get("LORA_ID") or "").strip()
LORA_FILE = (os.environ.get("LORA_FILE") or "").strip()
LORA_SCALE = clamp_float(os.environ.get("LORA_SCALE"), 1.0, 0.0, 2.0)
DEFAULT_STEPS_ENV = clamp_int(os.environ.get("DEFAULT_STEPS"), DEFAULT_STEPS, 1, 60)
DEFAULT_SIZE_ENV = clamp_int(os.environ.get("DEFAULT_SIZE"), DEFAULT_SIZE, 256, 1536)
DEFAULT_GUIDANCE_ENV = clamp_float(os.environ.get("DEFAULT_GUIDANCE"), DEFAULT_GUIDANCE, 0.0, 20.0)
MAX_CACHED_PIPELINES = clamp_int(os.environ.get("MAX_CACHED_PIPELINES"), 2, 1, 4)
ALLOW_DOWNLOAD = os.environ.get("ALLOW_DOWNLOAD", "1") not in ("0", "false", "False")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
MODEL_ROOT = resolve_root()

#: model reference -> {"pipe", "load_seconds", "model", "model_path", "model_type"}
_PIPELINES = {}
_PIPELINE_ORDER = []


def _log_preflight():
    log.info(
        "worker start: volume=%s model_root=%s cuda=%s device=%s",
        volume_available(), MODEL_ROOT,
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    )
    log.info("host model cache: root=%s repos=%s", HF_CACHE_ROOT, host_cache_repos())
    if MODEL_ID:
        log.info("default MODEL_ID=%s", MODEL_ID)


# --------------------------------------------------------------------------- #
# pipeline loading
# --------------------------------------------------------------------------- #
def _model_info(model_ref):
    info = ensure(model_ref, root=MODEL_ROOT, kind_hint="checkpoints",
                  allow_download=ALLOW_DOWNLOAD, token=HF_TOKEN)
    if info.get("error"):
        raise RuntimeError("model %s: %s" % (model_ref, info["error"]))
    return info


def _load_lora(pipe, lora_ref, weight, adapter_name):
    info = ensure(lora_ref, root=MODEL_ROOT, kind_hint="lora",
                  allow_download=ALLOW_DOWNLOAD, token=HF_TOKEN)
    if info.get("error"):
        raise RuntimeError("lora %s: %s" % (lora_ref, info["error"]))
    path = info["path"]
    if os.path.isfile(path):
        pipe.load_lora_weights(os.path.dirname(path), weight_name=os.path.basename(path),
                               adapter_name=adapter_name)
    else:  # a diffusers-format LoRA directory
        pipe.load_lora_weights(path, adapter_name=adapter_name)
    return {"ref": lora_ref, "weight": weight, "path": path, "cached": info["cached"],
            "download_seconds": info["download_seconds"]}


def _apply_vae(pipe, vae_ref):
    info = ensure(vae_ref, root=MODEL_ROOT, kind_hint="vae",
                  allow_download=ALLOW_DOWNLOAD, token=HF_TOKEN)
    if info.get("error"):
        raise RuntimeError("vae %s: %s" % (vae_ref, info["error"]))
    path = info["path"]
    if os.path.isdir(path):
        vae = AutoencoderKL.from_pretrained(path, torch_dtype=torch.float16)
    else:
        vae = AutoencoderKL.from_single_file(path, torch_dtype=torch.float16)
    pipe.vae = vae.to("cuda")
    return {"ref": vae_ref, "path": path, "cached": info["cached"]}


def _build_pipeline(model_ref, vae_ref, loras):
    """Load a pipeline from a *local* path produced by the model store."""
    model = _model_info(model_ref)
    t0 = time.time()
    if model["kind"] in ("hf_repo", "local_dir"):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model["path"], torch_dtype=torch.float16, use_safetensors=True,
            local_files_only=True, add_watermarker=False,
        )
    else:
        pipe = StableDiffusionXLPipeline.from_single_file(
            model["path"], torch_dtype=torch.float16, use_safetensors=True,
            add_watermarker=False,
        )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    entry = {
        "pipe": pipe,
        "model": model_ref,
        "model_path": model["path"],
        "model_type": model["kind"],
        "model_source": model.get("source"),
        "model_cached": model["cached"],
        "model_download_seconds": model["download_seconds"],
        "model_bytes": model["bytes"],
        "load_seconds": round(time.time() - t0, 2),
        "vae": None,
        "loras": [],
    }

    if vae_ref:
        entry["vae"] = _apply_vae(pipe, vae_ref)

    loras = list(loras)
    if not loras and LORA_ID and LORA_FILE:
        loras = [{"path": "%s::%s" % (LORA_ID, LORA_FILE), "weight": LORA_SCALE}]
        entry["loras_from_env"] = True
    loaded = []
    for idx, lora in enumerate(loras):
        loaded.append(_load_lora(pipe, lora["path"], lora["weight"], "lora_%d" % idx))
    if loaded:
        pipe.set_adapters(["lora_%d" % i for i in range(len(loaded))],
                          adapter_weights=[l["weight"] for l in loaded])
        entry["loras"] = loaded
    log.info("pipeline ready: %s (%s) load=%ss cached=%s loras=%d",
             model_ref, model["kind"], entry["load_seconds"], model["cached"], len(loaded))
    return entry


def get_pipeline(model_ref, vae_ref, loras):
    key = (model_ref, vae_ref or "", tuple(sorted((l["path"], l["weight"]) for l in loras)))
    if key in _PIPELINES:
        _PIPELINE_ORDER.remove(key)
        _PIPELINE_ORDER.append(key)
        entry = dict(_PIPELINES[key])
        entry["pipeline_reused"] = True
        return entry
    entry = _build_pipeline(model_ref, vae_ref, loras)
    entry["pipeline_reused"] = False
    _PIPELINES[key] = entry
    _PIPELINE_ORDER.append(key)
    while len(_PIPELINE_ORDER) > MAX_CACHED_PIPELINES:
        old = _PIPELINE_ORDER.pop(0)
        _PIPELINES.pop(old, None)
        torch.cuda.empty_cache()
        log.info("evicted pipeline from cache: %s", old[0])
    return dict(entry)


# --------------------------------------------------------------------------- #
# sampler / clip_skip
# --------------------------------------------------------------------------- #
def apply_sampler(pipe, sampler_name):
    if not sampler_name or sampler_name == DEFAULT_SAMPLER:
        return None
    spec = sampler_spec(sampler_name)  # raises ValueError on unknown names
    cls = getattr(importlib.import_module("diffusers"), spec["class"])
    try:
        pipe.scheduler = cls.from_config(pipe.scheduler.config, **spec["kwargs"])
        return None
    except Exception as exc:  # noqa: BLE001 - a bad scheduler must not kill the job
        return "%s: %r" % (sampler_name, exc)


def _make_encode_prompt(hidden_index):
    """SDXL prompt encoding with a configurable hidden-state index.

    diffusers hard-codes ``hidden_states[-2]`` (clip_skip=2). Re-installing this
    as an instance method is the documented way to change it without forking the
    pipeline. Only used when the caller asks for clip_skip != 2.
    """

    def encode_prompt(  # noqa: PLR0913 - mirrors the upstream signature
        self, prompt, prompt_2=None, device=None, num_images_per_prompt=1,
        do_classifier_free_guidance=True, negative_prompt=None, negative_prompt_2=None,
        prompt_embeds=None, negative_prompt_embeds=None, pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None, lora_scale=None, clip_skip=None,
    ):
        if prompt_embeds is not None:
            raise RuntimeError("clip_skip override cannot be combined with precomputed prompt_embeds")
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_2 = prompt_2 if prompt_2 is not None else prompt
        batch_size = len(prompt)
        text_encoders = [self.text_encoder, self.text_encoder_2]
        tokenizers = [self.tokenizer, self.tokenizer_2]

        prompt_embeds_list = []
        pooled_prompt_embeds = None
        for text_encoder, tokenizer, one_prompt in zip(text_encoders, tokenizers, [prompt, prompt_2]):
            text_input_ids = tokenizer(
                one_prompt, padding="max_length", max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids
            outputs = text_encoder(text_input_ids.to(device), output_hidden_states=True)
            pooled_prompt_embeds = outputs[0]
            prompt_embeds_list.append(outputs.hidden_states[hidden_index])
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1).to(
            dtype=self.text_encoder.dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1)

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = ([negative_prompt] * batch_size
                               if isinstance(negative_prompt, str) else negative_prompt)
            negative_prompt_2 = negative_prompt_2 if negative_prompt_2 is not None else negative_prompt
            negative_prompt_embeds_list = []
            for text_encoder, tokenizer, one_prompt in zip(text_encoders, tokenizers,
                                                           [negative_prompt, negative_prompt_2]):
                uncond_input_ids = tokenizer(
                    one_prompt, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt",
                ).input_ids
                outputs = text_encoder(uncond_input_ids.to(device), output_hidden_states=True)
                negative_pooled_prompt_embeds = outputs[0]
                negative_prompt_embeds_list.append(outputs.hidden_states[hidden_index])
            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1).to(
                dtype=self.text_encoder.dtype, device=device)
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
                1, num_images_per_prompt).view(batch_size * num_images_per_prompt, -1)
        else:
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    return encode_prompt


def apply_clip_skip(pipe, clip_skip):
    """Install/remove the clip_skip override; returns a warning string or None."""
    if clip_skip is None or clip_skip == 2:
        if "_clip_skip_original" in pipe.__dict__:
            pipe.encode_prompt = pipe.__dict__.pop("_clip_skip_original")
        return None
    original = pipe.__dict__.get("_clip_skip_original")
    if original is None:
        original = pipe.encode_prompt
        pipe._clip_skip_original = original
    if clip_skip == 1:
        hidden_index = -1
    else:
        hidden_index = -clip_skip
    pipe.encode_prompt = types.MethodType(_make_encode_prompt(hidden_index), pipe)
    return None


# --------------------------------------------------------------------------- #
# handler
# --------------------------------------------------------------------------- #
def handler(job):
    event = job or {}
    runpod.serverless.progress_update(event, "parsing request")
    req = parse_job_input(
        event.get("input"),
        default_steps=DEFAULT_STEPS_ENV,
        default_size=DEFAULT_SIZE_ENV,
        default_guidance=DEFAULT_GUIDANCE_ENV,
        default_model=MODEL_ID,
    )
    if req.get("ignored_keys"):
        log.info("ignoring legacy keys: %s", ",".join(req.pop("ignored_keys")))
    if not req["model"]:
        raise ValueError("no model requested and MODEL_ID is not set on the endpoint")

    request_t0 = time.time()
    runpod.serverless.progress_update(event, "loading model %s" % req["model"])
    entry = get_pipeline(req["model"], req["vae"], req["loras"])
    pipe = entry["pipe"]

    sampler_warning = apply_sampler(pipe, req["sampler"])
    if sampler_warning:
        log.warning("sampler fallback: %s", sampler_warning)
    clip_warning = apply_clip_skip(pipe, req["clip_skip"])
    if clip_warning:
        log.warning("clip_skip fallback: %s", clip_warning)

    seed = req["seed"]
    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    runpod.serverless.progress_update(event, "generating %dx%d @ %d steps" % (
        req["width"], req["height"], req["steps"]))
    t0 = time.time()
    result = pipe(
        prompt=req["prompt"],
        negative_prompt=req["negative_prompt"] or None,
        width=req["width"],
        height=req["height"],
        num_inference_steps=req["steps"],
        guidance_scale=req["guidance"],
        generator=generator,
    )
    generation_seconds = round(time.time() - t0, 2)

    buf = io.BytesIO()
    result.images[0].save(buf, format="PNG", optimize=False)
    png = buf.getvalue()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    gpu_total = (torch.cuda.get_device_properties(0).total_memory / 1024**3
                 if torch.cuda.is_available() else 0)
    meta = {
        "model": req["model"],
        "model_type": entry.get("model_type"),
        "model_path": entry.get("model_path"),
        "model_source": entry.get("model_source"),
        "model_cached": entry.get("model_cached"),
        "model_download_seconds": entry.get("model_download_seconds"),
        "pipeline_load_seconds": entry.get("load_seconds"),
        "pipeline_reused": entry.get("pipeline_reused"),
        "vae": entry.get("vae"),
        "loras": entry.get("loras") or [],
        "loras_from_env": entry.get("loras_from_env", False),
        "gpu_name": gpu_name,
        "gpu_total_memory_gb": round(gpu_total, 1),
        "generation_seconds": generation_seconds,
        "request_seconds": round(time.time() - request_t0, 2),
        "steps": req["steps"],
        "size": [req["width"], req["height"]],
        "guidance": req["guidance"],
        "sampler": req["sampler"],
        "sampler_warning": sampler_warning,
        "clip_skip": req["clip_skip"],
        "seed": seed,
        "image_bytes": len(png),
        "volume_available": volume_available(),
        "model_root": MODEL_ROOT,
    }
    log.info("done: %s %ss (gen %ss, load %ss, cached=%s) seed=%s",
             req["model"], meta["request_seconds"], generation_seconds,
             meta["pipeline_load_seconds"], meta["model_cached"], seed)
    return {"image": base64.b64encode(png).decode("ascii"), "meta": meta}


_log_preflight()

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
