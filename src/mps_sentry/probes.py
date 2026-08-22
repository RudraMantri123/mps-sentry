"""Differential probes that expose silent-correctness failures on Apple's MPS backend.

Each probe builds a scenario, runs it on MPS, and compares against a CPU reference
computed from the same inputs. A probe reports SILENT_CORRUPTION only when MPS
disagrees with CPU by more than float rounding can explain — the failure mode that
produces wrong numbers with no exception, warning, or NaN in the inputs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch

# fp16/bf16 accumulate differently on MPS; anything under these is drift, not corruption.
_DRIFT_TOLERANCE = {
    torch.float16: 5e-2,
    torch.bfloat16: 2e-1,
    torch.float32: 1e-4,
}


@dataclasses.dataclass
class ProbeResult:
    name: str
    status: str  # "ok" | "corrupt" | "error" | "skipped"
    detail: str = ""
    max_abs_diff: float | None = None
    affected: str | None = None

    @property
    def failed(self) -> bool:
        return self.status == "corrupt"


def _verdict(name: str, mps: torch.Tensor, cpu: torch.Tensor, dtype: torch.dtype, note: str = "") -> ProbeResult:
    mps_c = mps.detach().to("cpu", dtype=torch.float32)
    cpu_c = cpu.detach().to(torch.float32)
    nan_only_on_mps = int(torch.isnan(mps_c).sum() - torch.isnan(cpu_c).sum())
    finite = torch.isfinite(mps_c) & torch.isfinite(cpu_c)
    diff = (mps_c[finite] - cpu_c[finite]).abs().max().item() if finite.any() else 0.0
    tol = _DRIFT_TOLERANCE.get(dtype, 1e-4)

    if nan_only_on_mps > 0:
        return ProbeResult(name, "corrupt", f"{nan_only_on_mps} NaN present only on MPS. {note}", diff, "NaN")
    if diff > tol:
        return ProbeResult(name, "corrupt", f"values diverge beyond {dtype} rounding ({tol:g}). {note}", diff)
    return ProbeResult(name, "ok", note, diff)


def _poison_pool(shape: tuple[int, ...], dtype: torch.dtype, rounds: int = 2) -> None:
    """Fill and free MPS blocks of `shape` so the allocator can recycle NaN-bearing pages."""
    for _ in range(rounds):
        junk = torch.full(shape, float("nan"), device="mps", dtype=dtype)
        del junk


def probe_uninitialized_beta_zero(dtype: torch.dtype = torch.float16) -> ProbeResult:
    """`baddbmm(input=empty, beta=0)` must ignore `input` — including NaN in it.

    Exposed as all-black SDXL images in diffusers: huggingface/diffusers#14438.
    Upstream: pytorch/pytorch#187521 (fixed in torch >= 2.13).
    """
    b, t, d = 8, 1024, 64
    scores_shape = (b, t, t)
    q = torch.randn(b, t, d, device="mps", dtype=dtype)
    k = torch.randn(b, t, d, device="mps", dtype=dtype)

    _poison_pool(scores_shape, dtype)
    buf = torch.empty(scores_shape, device="mps", dtype=dtype)
    mps = torch.baddbmm(buf, q, k.transpose(-1, -2), beta=0, alpha=0.125)
    cpu = torch.baddbmm(
        torch.zeros(scores_shape, dtype=dtype), q.cpu(), k.cpu().transpose(-1, -2), beta=0, alpha=0.125
    )
    return _verdict("baddbmm(beta=0) ignores uninitialized input", mps, cpu, dtype,
                    "docs: input is ignored and nan/inf in it are not propagated")


def probe_addmm_beta_zero(dtype: torch.dtype = torch.float16) -> ProbeResult:
    """Same contract, `addmm` — checks whether the beta=0 guarantee holds family-wide."""
    n, m, k_ = 1024, 1024, 128
    a = torch.randn(n, k_, device="mps", dtype=dtype)
    b = torch.randn(k_, m, device="mps", dtype=dtype)

    _poison_pool((n, m), dtype)
    buf = torch.empty(n, m, device="mps", dtype=dtype)
    mps = torch.addmm(buf, a, b, beta=0, alpha=1.0)
    cpu = torch.addmm(torch.zeros(n, m, dtype=dtype), a.cpu(), b.cpu(), beta=0, alpha=1.0)
    return _verdict("addmm(beta=0) ignores uninitialized input", mps, cpu, dtype)


def probe_nonblocking_copy_freed_source(dtype: torch.dtype = torch.float32, rounds: int = 4) -> ProbeResult:
    """`to("mps", non_blocking=True)` must not read source storage freed before sync.

    Exposed as silently corrupted model weights in diffusers: huggingface/diffusers#13227.
    Upstream: pytorch/pytorch#189690.

    This failure is allocator-dependent, so it appears in only a few tensors per batch. The probe
    repeats the scenario up to `rounds` times and reports the first round that corrupts; a clean
    result lowers the probability of the bug but does not prove its absence.
    """
    count, size = 64, 2048
    for attempt in range(rounds):
        refs, moved = [], []
        for i in range(count):
            src = torch.randn(size, size, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(i))
            refs.append(src.to(dtype))
            tmp = src.to(dtype)                                # short-lived cast temporary
            moved.append(tmp.to("mps", non_blocking=True))     # async copy; tmp dies next iteration
            del tmp, src
        torch.mps.synchronize()                                # single sync, at the very end

        bad = [i for i, (m, r) in enumerate(zip(moved, refs)) if not torch.equal(m.cpu(), r)]
        if bad:
            worst = max((m.cpu().float() - r.float()).abs().max().item() for m, r in zip(moved, refs))
            return ProbeResult(
                "non_blocking H2D copy from freed source", "corrupt",
                f"{len(bad)}/{count} tensors read released storage (round {attempt + 1}/{rounds})",
                worst, f"tensors {bad[:6]}",
            )
        del refs, moved
        torch.mps.empty_cache()

    return ProbeResult(
        "non_blocking H2D copy from freed source", "ok",
        f"clean across {rounds} rounds of {count} tensors (allocator-dependent; not proof of absence)",
        0.0,
    )


def probe_float64_support() -> ProbeResult:
    """Informational: MPS has no float64, so hardcoded float64 raises instead of corrupting."""
    try:
        torch.arange(4, dtype=torch.float64, device="mps")
        return ProbeResult("float64 tensors", "ok", "float64 accepted on this backend")
    except TypeError as exc:
        return ProbeResult("float64 tensors", "skipped", f"unsupported as expected: {str(exc)[:60]}")


def probe_reduction_numerics(dtype: torch.dtype = torch.float32) -> ProbeResult:
    """Large-reduction numerics: softmax over a long axis, a known MPS trouble spot."""
    x = torch.randn(4, 4096, 4096, dtype=dtype)
    mps = torch.softmax(x.to("mps"), dim=-1)
    cpu = torch.softmax(x, dim=-1)
    return _verdict("softmax over long axis", mps, cpu, dtype)


def probe_layernorm_numerics(dtype: torch.dtype = torch.float16) -> ProbeResult:
    """LayerNorm in low precision over a wide feature axis."""
    x = torch.randn(2, 2048, 4096, dtype=dtype)
    w = torch.randn(4096, dtype=dtype)
    b = torch.randn(4096, dtype=dtype)
    mps = torch.nn.functional.layer_norm(x.to("mps"), (4096,), w.to("mps"), b.to("mps"))
    cpu = torch.nn.functional.layer_norm(x, (4096,), w, b)
    return _verdict("layer_norm over wide axis", mps, cpu, dtype)


ALL_PROBES: dict[str, Callable[[], ProbeResult]] = {
    "baddbmm-beta0": probe_uninitialized_beta_zero,
    "addmm-beta0": probe_addmm_beta_zero,
    "nonblocking-copy": probe_nonblocking_copy_freed_source,
    "float64": probe_float64_support,
    "softmax": probe_reduction_numerics,
    "layernorm": probe_layernorm_numerics,
}
