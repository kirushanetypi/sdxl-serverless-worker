"""Pure helpers for the custom SDXL worker.

Everything in here is deliberately torch-free / diffusers-free so it can be
unit-tested on a laptop or the VPS without a GPU.

Storage model (decided in kanban t_970eafc9, supersedes the earlier
"RunPod Model Caching" draft):

  * models live on a RunPod **network volume** mounted at ``/runpod-volume``;
  * layout:

        /runpod-volume/models/checkpoints/   # diffusers dirs and single-file .safetensors
        /runpod-volume/models/loras/         # LoRA .safetensors
        /runpod-volume/models/vae/           # VAE .safetensors / diffusers dirs
        /runpod-volume/models/registry.json  # bookkeeping (what was fetched, when, how big)

  * cache-on-first-use: a model that is not on the volume yet is downloaded by
    the worker on the first request that asks for it and then reused forever.

RunPod's own "Model Caching" (endpoint **Model** field, set through the console
or ``runpodctl serverless update --model-reference``) is used *in addition*, for
the single checkpoint that endpoint serves most: the preloaded copy on the
worker host is preferred by ``model_store.ensure`` when it is complete, because
it reads much faster than the same bytes streamed off a network volume and the
download itself is not billed. It does not replace the volume store - the cache
holds exactly one repo per endpoint, so every other checkpoint still comes from
``/runpod-volume/models``.
"""
import os

#: Mount point of a RunPod network volume inside Serverless workers.
VOLUME_ROOT = "/runpod-volume"

#: Root of the model store on the volume.
MODEL_ROOT = os.path.join(VOLUME_ROOT, "models")

CHECKPOINTS_DIRNAME = "checkpoints"
LORAS_DIRNAME = "loras"
VAE_DIRNAME = "vae"
REGISTRY_FILENAME = "registry.json"

#: Fallback store used when no network volume is attached (non-persistent; the
#: worker still works, it just re-downloads on the next cold start).
FALLBACK_MODEL_ROOT = "/tmp/nsfw-models"

#: Default checkpoint. Deliberately empty in the repo: the concrete model is set
#: through the endpoint's MODEL_ID env var, so no model list is committed here.
DEFAULT_MODEL_ID = ""

MAX_STEPS = 60
MAX_SIDE = 1536
MAX_LORA_WEIGHT = 2.0
MAX_CACHED_PIPELINES_ENV = "MAX_CACHED_PIPELINES"

DEFAULT_STEPS = 30
DEFAULT_SIZE = 1024
DEFAULT_GUIDANCE = 6.5
DEFAULT_CLIP_SKIP = 2  # diffusers' own SDXL default (hidden_states[-2])

#: Only used when the caller does not send a negative prompt at all.
DEFAULT_NEGATIVE = (
    "blurry, low quality, deformed, ugly, extra limbs, extra fingers, bad anatomy, "
    "watermark, text, signature"
)

#: Tag prefix for Illustrious/Pony-family checkpoints. NOT applied by default
#: (a realistic SDXL checkpoint does not understand score_* tags); pass
#: ``quality_prefix`` explicitly, or send ``appearance``/``style_positive``.
DEFAULT_QUALITY_PREFIX = "score_9, score_8_up, score_7_up"

DEFAULT_SAMPLER = "auto"

#: name -> diffusers scheduler class + constructor kwargs.
SAMPLERS = {
    "auto": None,  # keep the scheduler the checkpoint ships with
    "euler": {"class": "EulerDiscreteScheduler", "kwargs": {}},
    "euler_a": {"class": "EulerAncestralDiscreteScheduler", "kwargs": {}},
    "dpmpp_2m": {
        "class": "DPMSolverMultistepScheduler",
        "kwargs": {"algorithm_type": "dpmsolver++", "use_karras_sigmas": False},
    },
    "dpmpp_2m_karras": {
        "class": "DPMSolverMultistepScheduler",
        "kwargs": {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True},
    },
    "dpmpp_2m_sde": {
        "class": "DPMSolverMultistepScheduler",
        "kwargs": {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    },
    "dpmpp_sde": {"class": "DPMSolverSDEScheduler", "kwargs": {}},
    "ddim": {"class": "DDIMScheduler", "kwargs": {}},
    "pndm": {"class": "PNDMScheduler", "kwargs": {}},
    "lms": {"class": "LMSDiscreteScheduler", "kwargs": {}},
    "heun": {"class": "HeunDiscreteScheduler", "kwargs": {}},
    "unipc": {"class": "UniPCMultistepScheduler", "kwargs": {}},
}


def clamp_int(value, default, lo, hi):
    """Coerce ``value`` to int inside [lo, hi]; fall back to ``default`` on junk."""
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, out))


def clamp_float(value, default, lo, hi):
    """Coerce ``value`` to float inside [lo, hi]; fall back to ``default`` on junk."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, out))


def align_to_8(value):
    """SDXL wants dimensions that are a multiple of 8."""
    return value - (value % 8)


def build_prompt(prompt, appearance="", style_positive="", quality_prefix=""):
    """Join optional prompt fragments in a stable order (empty parts dropped)."""
    parts = [quality_prefix or "", appearance or "", (prompt or "").strip(), style_positive or ""]
    return ", ".join(p for p in parts if p and p.strip())


def sampler_spec(name):
    """Return the scheduler spec for ``name`` or raise ValueError."""
    if not name:
        raise ValueError("empty sampler name")
    key = str(name).strip().lower().replace("-", "_")
    if key not in SAMPLERS:
        raise ValueError(
            "unknown sampler %r (known: %s)" % (name, ", ".join(sorted(SAMPLERS)))
        )
    return SAMPLERS[key]


def parse_loras(raw, single=None, single_weight=None):
    """Normalise the several accepted LoRA spellings into ``[{path, weight}, ...]``.

    Accepted:
      * ``loras``: list of dicts ``{"path": ..., "weight": 0.8}``
      * ``loras``: list of strings (``"repo/file.safetensors"``, ``"repo::file"``,
        volume path, URL)
      * ``lora`` + optional ``lora_weight``
    """
    out = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, str):
                out.append({"path": item.strip(), "weight": 1.0})
            elif isinstance(item, dict):
                path = item.get("path") or item.get("name") or item.get("file") or ""
                if not str(path).strip():
                    continue
                out.append({
                    "path": str(path).strip(),
                    "weight": clamp_float(item.get("weight", 1.0), 1.0, 0.0, MAX_LORA_WEIGHT),
                })
    elif isinstance(raw, dict):
        return parse_loras([raw])
    elif isinstance(raw, str) and raw.strip():
        out.append({"path": raw.strip(), "weight": 1.0})

    if single and str(single).strip():
        out.append({
            "path": str(single).strip(),
            "weight": clamp_float(single_weight, 1.0, 0.0, MAX_LORA_WEIGHT),
        })
    return [lora for lora in out if lora["path"]]


def parse_job_input(
    raw,
    default_steps=DEFAULT_STEPS,
    default_size=DEFAULT_SIZE,
    default_guidance=DEFAULT_GUIDANCE,
    default_model="",
):
    """Normalise ``event['input']`` from a RunPod job into a generation request.

    Canonical input shape (kanban t_970eafc9)::

        {"model": "<hf repo | repo::file | url | volume path>",
         "prompt": "...", "negative_prompt": "...",
         "width": 1024, "height": 1024, "steps": 30, "guidance": 6.5,
         "seed": 12345, "sampler": "dpmpp_2m", "clip_skip": 2,
         "vae": "<hf repo | url | volume path>",
         "loras": [{"path": "...", "weight": 0.8}]}

    Backwards compatibility with the payload the bot already sends
    (``num_inference_steps``, ``guidance_scale``, ``refiner_inference_steps``,
    ``high_noise_frac``, ``appearance``, ``style_positive``) is kept on purpose.
    """
    raw = raw if isinstance(raw, dict) else {}
    prompt = (raw.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("input.prompt is required")

    fragments = {
        "appearance": raw.get("appearance", ""),
        "style_positive": raw.get("style_positive", ""),
        "quality_prefix": raw.get("quality_prefix", ""),
    }
    if any(str(v or "").strip() for v in fragments.values()):
        prompt = build_prompt(prompt, **fragments)

    height = align_to_8(clamp_int(raw.get("height"), default_size, 256, MAX_SIDE))
    width = align_to_8(clamp_int(raw.get("width"), default_size, 256, MAX_SIDE))

    steps_raw = raw.get("steps", raw.get("num_inference_steps"))
    guidance_raw = raw.get("guidance", raw.get("guidance_scale"))

    model = str(raw.get("model") or raw.get("model_id") or raw.get("checkpoint") or
                default_model or "").strip()
    vae = str(raw.get("vae") or raw.get("vae_path") or "").strip()

    req = {
        "prompt": prompt,
        "negative_prompt": (raw["negative_prompt"].strip()
                            if isinstance(raw.get("negative_prompt"), str) else
                            (DEFAULT_NEGATIVE if "negative_prompt" not in raw else "")),
        "width": width,
        "height": height,
        "steps": clamp_int(steps_raw, default_steps, 1, MAX_STEPS),
        "guidance": clamp_float(guidance_raw, default_guidance, 0.0, 20.0),
        "seed": (clamp_int(raw.get("seed"), None, -1, 2**31 - 1)
                 if raw.get("seed") is not None else None),
        "sampler": (str(raw.get("sampler") or DEFAULT_SAMPLER).strip().lower()
                    or DEFAULT_SAMPLER),
        "clip_skip": clamp_int(raw.get("clip_skip"), DEFAULT_CLIP_SKIP, 1, 4),
        "model": model,
        "vae": vae,
        "loras": parse_loras(raw.get("loras"), raw.get("lora"), raw.get("lora_weight")),
    }
    ignored = [k for k in ("refiner_inference_steps", "high_noise_frac") if k in raw]
    if ignored:
        req["ignored_keys"] = ignored
    return req
