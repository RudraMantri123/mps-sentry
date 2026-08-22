"""mps-sentry — differential correctness checks for PyTorch's Apple Silicon (MPS) backend."""

from .probes import ALL_PROBES, ProbeResult

__all__ = ["ALL_PROBES", "ProbeResult", "mps_usable", "run", "__version__"]
__version__ = "0.1.1"


def mps_usable() -> tuple[bool, str]:
    """Whether MPS is not just present but actually able to allocate and compute.

    ``torch.backends.mps.is_available()`` returns True in some virtualised environments —
    notably GitHub's macOS runners — where every allocation then fails with
    "MPS backend out of memory (MPS allocated: 0 bytes)". Checking the flag alone is not
    enough, so this attempts a real allocation and a real kernel launch.
    """
    import torch

    if not torch.backends.mps.is_available():
        return False, "torch.backends.mps.is_available() is False"
    try:
        probe = torch.ones(64, 64, device="mps")
        _ = (probe @ probe).sum().item()
        del probe
        torch.mps.empty_cache()
    except Exception as exc:  # noqa: BLE001 - any failure here means unusable
        return False, f"MPS reports available but cannot compute: {type(exc).__name__}: {exc}"
    return True, "ok"


def run(names=None):
    """Run probes and return a list of ProbeResult. `names` filters by probe key."""
    import torch

    usable, reason = mps_usable()
    if not usable:
        raise RuntimeError(reason)

    selected = ALL_PROBES if names is None else {k: v for k, v in ALL_PROBES.items() if k in names}
    results = []
    for key, probe in selected.items():
        try:
            results.append(probe())
        except Exception as exc:  # a probe crashing is itself a finding worth surfacing
            results.append(ProbeResult(key, "error", f"{type(exc).__name__}: {exc}"))
        finally:
            torch.mps.empty_cache()
    return results
