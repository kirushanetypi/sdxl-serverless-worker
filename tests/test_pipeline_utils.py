"""Unit tests for the pure (torch-free) worker helpers.

Run:  /root/nsfw-bot/bot/venv/bin/python -m pytest -q image-endpoint/worker-image/tests
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_utils import (  # noqa: E402
    DEFAULT_NEGATIVE,
    align_to_8,
    build_prompt,
    clamp_float,
    clamp_int,
    grey_health,
    parse_job_input,
    parse_loras,
    sampler_spec,
)


# --------------------------------------------------------------------------- #
# generated-frame sanity check (kanban t_06eb5e83)
# --------------------------------------------------------------------------- #
def _noise_pixels(n=128 * 128, width=128, seed=7):
    """Deterministic pseudo-noise, like a frame from a broken checkpoint/VAE."""
    out = []
    state = seed
    for _ in range(n):
        state = (1103515245 * state + 12345) % (2 ** 31)
        out.append((state >> 16) % 256)
    return out


def _picture_pixels(n=128 * 128, width=128):
    """A structured frame: smooth gradients and a few flat areas."""
    out = []
    for i in range(n):
        row, col = divmod(i, width)
        out.append((row * 2 + col // 8) % 256 if (row // 16 + col // 16) % 2 else 40)
    return out


def test_grey_health_flags_noise():
    got = grey_health(_noise_pixels(), width=128)
    assert got["suspected_noise"] is True
    assert got["ok"] is False
    assert got["stddev"] >= 55          # white noise is ~74
    assert got["neighbour_diff"] >= 30  # ~42 for independent samples


def test_grey_health_accepts_a_structured_frame():
    got = grey_health(_picture_pixels(), width=128)
    assert got["suspected_noise"] is False
    assert got["ok"] is True
    assert got["unique_values"] > 1


def test_grey_health_flags_a_flat_frame_as_blank():
    got = grey_health([0] * (128 * 128), width=128)
    assert got["suspected_blank"] is True and got["ok"] is False
    assert got["stddev"] == 0.0 and got["neighbour_diff"] == 0.0


def test_grey_health_handles_empty_input():
    got = grey_health([], width=128)
    assert got["pixels"] == 0 and got["ok"] is False


def test_grey_health_ignores_row_wraps_when_measuring_neighbours():
    # a horizontal edge is structure: one big difference per row, not per pixel
    width, height = 128, 128
    got = grey_health([0 if (i % width) < width // 2 else 255
                       for i in range(width * height)], width=width)
    assert got["neighbour_diff"] == pytest.approx(255 / 127, abs=0.1)
    assert got["suspected_noise"] is False


def test_clamp_helpers():
    assert clamp_int("12", 8, 1, 50) == 12
    assert clamp_int("junk", 8, 1, 50) == 8
    assert clamp_int(999, 8, 1, 50) == 50
    assert clamp_int(None, 8, 1, 50) == 8
    assert clamp_float("6.5", 6.5, 0, 20) == 6.5
    assert clamp_float("junk", 6.5, 0, 20) == 6.5
    assert clamp_float(-3, 6.5, 0, 20) == 0


def test_align_to_8():
    assert align_to_8(1023) == 1016
    assert align_to_8(1024) == 1024
    assert align_to_8(2001) == 2000


def test_build_prompt_is_verbatim_without_fragments():
    assert build_prompt("a red apple") == "a red apple"
    assert build_prompt("  a red apple  ") == "a red apple"


def test_build_prompt_orders_fragments_and_drops_empties():
    got = build_prompt("stockings, on bed", appearance="red hair",
                       style_positive="anime style", quality_prefix="score_9")
    assert got == "score_9, red hair, stockings, on bed, anime style"
    assert build_prompt("solo", appearance=None, style_positive="") == "solo"


def test_sampler_spec_known_and_unknown():
    assert sampler_spec("dpmpp_2m")["class"] == "DPMSolverMultistepScheduler"
    assert sampler_spec("DPMPP_2M_KARRAS")["kwargs"]["use_karras_sigmas"] is True
    assert sampler_spec("auto") is None
    with pytest.raises(ValueError):
        sampler_spec("not-a-sampler")
    with pytest.raises(ValueError):
        sampler_spec("")


def test_parse_loras_all_spellings():
    assert parse_loras(None) == []
    assert parse_loras("org/repo::l.safetensors") == [
        {"path": "org/repo::l.safetensors", "weight": 1.0}]
    assert parse_loras([{"path": "a.safetensors", "weight": 0.8}]) == [
        {"path": "a.safetensors", "weight": 0.8}]
    assert parse_loras(["a.safetensors", "  "]) == [{"path": "a.safetensors", "weight": 1.0}]
    assert parse_loras(None, single="b.safetensors", single_weight=2) == [
        {"path": "b.safetensors", "weight": 2.0}]
    # weight is clamped, junk entries are dropped
    assert parse_loras([{"path": "a", "weight": 99}]) == [{"path": "a", "weight": 2.0}]
    assert parse_loras([{"path": "  "}]) == []


def test_parse_job_input_canonical_contract():
    out = parse_job_input({
        "model": "org/repo::model.safetensors",
        "prompt": "a red apple",
        "negative_prompt": "blurry",
        "width": 1024,
        "height": 1024,
        "steps": 8,
        "guidance": 1.5,
        "seed": 42,
        "sampler": "dpmpp_2m",
        "clip_skip": 1,
        "vae": "org/vae.safetensors",
        "loras": [{"path": "org/repo::lightning.safetensors", "weight": 0.9}],
    })
    assert out["prompt"] == "a red apple"          # verbatim, no score_* prefix
    assert out["negative_prompt"] == "blurry"
    assert (out["width"], out["height"]) == (1024, 1024)
    assert out["steps"] == 8
    assert out["guidance"] == 1.5
    assert out["seed"] == 42
    assert out["sampler"] == "dpmpp_2m"
    assert out["clip_skip"] == 1
    assert out["model"] == "org/repo::model.safetensors"
    assert out["vae"] == "org/vae.safetensors"
    assert out["loras"] == [{"path": "org/repo::lightning.safetensors", "weight": 0.9}]
    assert "ignored_keys" not in out


def test_parse_job_input_keeps_bot_payload_working():
    """bot/generation.py sends num_inference_steps/guidance_scale + refiner keys."""
    raw = {
        "prompt": "woman in black stockings",
        "negative_prompt": "blurry",
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 6.5,
        "refiner_inference_steps": 18,
        "high_noise_frac": 0.8,
    }
    out = parse_job_input(raw)
    assert out["prompt"] == "woman in black stockings"
    assert out["steps"] == 50
    assert out["guidance"] == 6.5
    assert out["model"] == ""            # falls back to the endpoint's MODEL_ID
    assert sorted(out["ignored_keys"]) == ["high_noise_frac", "refiner_inference_steps"]


def test_parse_job_input_applies_optional_prompt_fragments():
    out = parse_job_input({
        "prompt": "stockings",
        "appearance": "red hair",
        "style_positive": "anime style",
        "quality_prefix": "score_9",
    })
    assert out["prompt"] == "score_9, red hair, stockings, anime style"


def test_parse_job_input_defaults_and_clamps():
    out = parse_job_input({"prompt": "x", "width": 1023, "height": 2000,
                           "steps": 999, "guidance": "junk", "clip_skip": 99})
    assert out["width"] == 1016
    assert out["height"] == 1536
    assert out["steps"] == 60
    assert out["guidance"] == 6.5
    assert out["clip_skip"] == 4
    assert out["sampler"] == "auto"
    assert out["seed"] is None
    assert out["negative_prompt"] == DEFAULT_NEGATIVE


def test_parse_job_input_explicit_empty_negative_is_respected():
    out = parse_job_input({"prompt": "x", "negative_prompt": ""})
    assert out["negative_prompt"] == ""


def test_parse_job_input_model_alias_and_default():
    assert parse_job_input({"prompt": "x", "model_id": "org/repo"})["model"] == "org/repo"
    assert parse_job_input({"prompt": "x", "checkpoint": "org/repo"})["model"] == "org/repo"
    assert parse_job_input({"prompt": "x"}, default_model="fallback/repo")["model"] == "fallback/repo"


def test_parse_job_input_requires_prompt():
    with pytest.raises(ValueError):
        parse_job_input({})
    with pytest.raises(ValueError):
        parse_job_input({"prompt": "   "})
    with pytest.raises(ValueError):
        parse_job_input(None)
