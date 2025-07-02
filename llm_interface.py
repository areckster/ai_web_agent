# llm_interface.py  — quiet, no Metal init spam (2025-06-18)
from __future__ import annotations

import contextlib
import os
from llama_cpp import Llama

# ─────────────────────────── globals ────────────────────────────
os.environ["LLAMA_CPP_LOG_LEVEL"] = "ERROR"           # silence C-side logs
_MODEL_PATH = "./openhermes-2.5-mistral-7b.Q3_K_M.gguf"
 
# ─────────────────────── model initialisation ──────────────────
with open(os.devnull, "w") as _null, contextlib.redirect_stderr(_null):
    _llm = Llama(
        model_path=_MODEL_PATH,
        n_ctx=8192,
        n_gpu_layers=-1,        # full GPU offload on Apple Silicon
        verbose=False,
    )

try:                           # (for llama_cpp ≥ 0.2.80)
    _llm.set_print_timings(False)
except AttributeError:
    pass

# ──────────────────────── internal helpers ─────────────────────
def _budget(prompt: str, reserve_ctx: int, hard_cap: int | None) -> int:
    used = len(_llm.tokenize(prompt.encode()))
    avail = _llm.n_ctx() - used - reserve_ctx
    if hard_cap is not None:
        avail = min(avail, hard_cap)
    return max(64, avail)

def _call_block(prompt: str,
                temperature: float = 0.32,
                reserve_ctx: int = 400,
                hard_cap: int | None = None) -> str:
    out = _llm(
        prompt,
        max_tokens=_budget(prompt, reserve_ctx, hard_cap),
        temperature=temperature,
        top_p=0.8,
        repeat_penalty=1.15,
        stop=["Observation:"],
    )
    return out["choices"][0]["text"].strip()

def _call_stream(prompt: str,
                 temperature: float = 0.32,
                 reserve_ctx: int = 400,
                 hard_cap: int | None = None) -> str:
    out = ""
    for chunk in _llm(
        prompt,
        max_tokens=_budget(prompt, reserve_ctx, hard_cap),
        temperature=temperature,
        top_p=0.8,
        repeat_penalty=1.15,
        stop=["Observation:"],
        stream=True,                     # ← key flag
    ):
        tok = chunk["choices"][0]["text"]
        print(tok, end="", flush=True)   # real-time console display
        out += tok
    return out.strip()

# ───────────────────────── public API ──────────────────────────
def call_llm(prompt: str,
             self_consistency: int = 1,
             stream: bool = False) -> str:
    """
    Parameters
    ----------
    prompt : str
        Full prompt (system + user + history).
    self_consistency : int, optional
        If > 1, generates multiple answers and takes majority vote.
        Streaming is disabled in that mode.
    stream : bool, optional
        If True, tokens are printed to stdout as soon as they appear.
    """
    if self_consistency <= 1:
        return _call_stream(prompt) if stream else _call_block(prompt)

    # self-consistency not compatible with streaming – fall back.
    outs = [
        _call_block(prompt, temperature=0.20 + 0.05 * i)
        for i in range(self_consistency)
    ]
    return max(set(outs), key=outs.count)

def call_llm_long(prompt: str,
                  self_consistency: int = 1,
                  hard_cap: int | None = 2048,
                  stream: bool = False) -> str:
    """
    Longer generation helper (uses smaller reservation).
    """
    def _one(temp: float, do_stream: bool) -> str:
        func = _call_stream if do_stream else _call_block
        return func(prompt, temperature=temp,
                    reserve_ctx=256, hard_cap=hard_cap)

    if self_consistency <= 1:
        return _one(0.20, stream)

    outs = [_one(0.20 + 0.05 * i, False) for i in range(self_consistency)]
    return max(set(outs), key=outs.count)