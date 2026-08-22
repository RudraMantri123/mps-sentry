"""mps-sentry — differential correctness checks for PyTorch's Apple Silicon (MPS) backend."""

from .probes import ALL_PROBES, ProbeResult

__all__ = ["ALL_PROBES", "ProbeResult", "run", "__version__"]
__version__ = "0.1.0"


def run(names=None):
    """Run probes and return a list of ProbeResult. `names` filters by probe key."""
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine")
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
