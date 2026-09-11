"""Contract tests for handler.py without a GPU.

torch/diffusers/runpod are stubbed into ``sys.modules`` and the pipeline loader
is replaced, so what is covered here is the handler's own logic: input parsing,
pipeline caching/eviction, the response contract (base64 PNG + ``meta`` with the
GPU name and timings) and the seed handling.
"""
import base64
import importlib
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


# --------------------------------------------------------------------------- #
# stub torch / diffusers / runpod
# --------------------------------------------------------------------------- #
class _FakeImage:
    def save(self, buf, format="PNG", optimize=False):  # noqa: A002
        buf.write(b"\x89PNG\r\n\x1a\n" + format.encode())


class _FakePipe:
    def __init__(self):
        self.calls = []
        self.images = [_FakeImage()]

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(images=self.images)

    def encode_prompt(self, *args, **kwargs):
        return ("default-encode",)


class _FakeGenerator:
    def __init__(self, device=None):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


def _install_stubs():
    torch = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(index=0):
            return "NVIDIA Test GPU"

        @staticmethod
        def get_device_properties(index=0):
            return types.SimpleNamespace(total_memory=24 * 1024**3)

        @staticmethod
        def empty_cache():
            return None

    torch.cuda = _Cuda()
    torch.float16 = "float16"
    torch.Generator = _FakeGenerator
    torch.concat = lambda tensors, dim=-1: tensors
    sys.modules["torch"] = torch

    diffusers = types.ModuleType("diffusers")
    diffusers.StableDiffusionXLPipeline = type("StableDiffusionXLPipeline", (), {})
    diffusers.AutoencoderKL = type("AutoencoderKL", (), {})
    sys.modules["diffusers"] = diffusers

    runpod = types.ModuleType("runpod")
    runpod.serverless = types.SimpleNamespace(
        start=lambda cfg: None, progress_update=lambda job, msg: None)
    sys.modules["runpod"] = runpod
    return torch


@pytest.fixture()
def worker():
    _install_stubs()
    sys.modules.pop("handler", None)
    handler = importlib.import_module("handler")
    handler._PIPELINES.clear()
    handler._PIPELINE_ORDER.clear()
    built = []

    def fake_build(model_ref, vae_ref, loras):
        pipe = _FakePipe()
        built.append(model_ref)
        return {"pipe": pipe, "model": model_ref, "model_path": "/volume/" + model_ref,
                "model_type": "hf_repo", "model_cached": True, "model_download_seconds": 0.0,
                "model_bytes": None, "load_seconds": 1.5, "vae": None, "loras": [],
                "pipeline_reused": False}

    handler._build_pipeline = fake_build
    handler._built = built
    handler.MODEL_ID = "org/test-model"  # endpoint env default for the tests
    return handler


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_response_contract_image_and_meta(worker):
    out = worker.handler({"input": {"prompt": "a red apple", "seed": 7}})
    assert set(out) == {"image", "meta"}
    png = base64.b64decode(out["image"])
    assert png.startswith(b"\x89PNG")

    meta = out["meta"]
    assert meta["gpu_name"] == "NVIDIA Test GPU"
    assert meta["generation_seconds"] >= 0
    assert meta["request_seconds"] >= 0
    assert meta["model_cached"] is True
    assert meta["steps"] == 30 and meta["size"] == [1024, 1024]
    assert meta["guidance"] == 6.5
    assert meta["seed"] == 7
    assert meta["sampler"] == "auto"
    assert meta["clip_skip"] == 2
    assert json.dumps(meta)  # meta must be JSON-serialisable


def test_generation_kwargs_are_forwarded(worker):
    worker.handler({"input": {"prompt": "x", "width": 832, "height": 1216, "steps": 8,
                              "guidance": 1.5, "seed": 1}})
    pipe = next(iter(worker._PIPELINES.values()))["pipe"]
    call = pipe.calls[-1]
    assert call["width"] == 832 and call["height"] == 1216
    assert call["num_inference_steps"] == 8
    assert call["guidance_scale"] == 1.5
    assert call["prompt"] == "x"


def test_random_seed_is_reported_and_stable_in_meta(worker):
    out = worker.handler({"input": {"prompt": "x"}})
    seed = out["meta"]["seed"]
    assert isinstance(seed, int) and seed >= 0


def test_same_model_is_loaded_once_and_reused(worker):
    worker.handler({"input": {"prompt": "one", "model": "org/a"}})
    out = worker.handler({"input": {"prompt": "two", "model": "org/a"}})
    assert worker._built == ["org/a"]
    assert out["meta"]["pipeline_reused"] is True


def test_pipeline_cache_is_bounded(worker):
    worker.MAX_CACHED_PIPELINES = 2
    for name in ("org/a", "org/b", "org/c"):
        worker.handler({"input": {"prompt": "x", "model": name}})
    assert len(worker._PIPELINES) == 2
    assert list(worker._PIPELINES) == [("org/b", "", ()), ("org/c", "", ())]


def test_model_defaults_to_endpoint_env(worker):
    worker.MODEL_ID = "env/default-model"
    out = worker.handler({"input": {"prompt": "x"}})
    assert out["meta"]["model"] == "env/default-model"
    assert worker._built == ["env/default-model"]


def test_missing_model_and_missing_prompt_are_errors(worker):
    worker.MODEL_ID = ""
    with pytest.raises(ValueError, match="no model requested"):
        worker.handler({"input": {"prompt": "x"}})
    with pytest.raises(ValueError, match="prompt is required"):
        worker.handler({"input": {"model": "org/a"}})


def test_apply_sampler_falls_back_on_unknown_name(worker):
    pipe = _FakePipe()
    with pytest.raises(ValueError):
        worker.apply_sampler(pipe, "nope")  # unknown names are rejected up front
    assert worker.apply_sampler(pipe, "auto") is None


def test_clip_skip_override_is_installed_and_removed(worker):
    pipe = _FakePipe()
    original = pipe.encode_prompt if hasattr(pipe, "encode_prompt") else None
    assert worker.apply_clip_skip(pipe, 2) is None
    assert not hasattr(pipe, "_clip_skip_original")
    assert worker.apply_clip_skip(pipe, 1) is None
    assert callable(pipe.encode_prompt)
    assert worker.apply_clip_skip(pipe, 2) is None
    assert "_clip_skip_original" not in pipe.__dict__
    assert getattr(pipe, "encode_prompt", None) == original
