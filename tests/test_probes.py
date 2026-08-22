import pytest
import torch

import mps_sentry
from mps_sentry.probes import ALL_PROBES

requires_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")


@requires_mps
@pytest.mark.parametrize("name", sorted(ALL_PROBES))
def test_probe_returns_a_verdict(name):
    result = ALL_PROBES[name]()
    assert result.status in {"ok", "corrupt", "error", "skipped"}
    assert result.name


@requires_mps
def test_run_covers_every_probe():
    assert len(mps_sentry.run()) == len(ALL_PROBES)


@requires_mps
def test_cpu_reference_agrees_with_itself():
    # Sanity check on the tolerance logic: identical tensors must never be reported as corrupt.
    from mps_sentry.probes import _verdict

    x = torch.randn(64, 64, dtype=torch.float16)
    assert not _verdict("identity", x.to("mps"), x, torch.float16).failed
