# mps-sentry

Differential correctness checks for PyTorch's Apple Silicon (MPS) backend.

MPS has a class of bugs that produce **wrong numbers with no exception, no warning, and no NaN in
your inputs**. They surface as black images, garbage outputs, or a model that is quietly a little
worse than it should be — and because nothing crashes, they are usually blamed on the model.

`mps-sentry` runs each check on MPS and against a CPU reference built from the same inputs, then
flags only disagreement that float rounding cannot explain.

```console
$ pip install mps-sentry
$ mps-sentry

mps-sentry 0.1.0
torch 2.12.0 | macOS 26.4.1 | arm64

[CORRUPT] baddbmm(beta=0) ignores uninitialized input  max|Δ|=0.000e+00
           8388608 NaN present only on MPS. docs: input is ignored and nan/inf in it are not propagated
[  ok  ] addmm(beta=0) ignores uninitialized input     max|Δ|=1.562e-02
[CORRUPT] non_blocking H2D copy from freed source       max|Δ|=7.641e+00
           2/64 tensors read released storage (round 2/4)
[ skip ] float64 tensors
[  ok  ] softmax over long axis                        max|Δ|=1.490e-08
[  ok  ] layer_norm over wide axis                     max|Δ|=3.906e-03

2 silent-correctness failure(s) on this torch build
```

Exit code is `1` when a silent-correctness failure is found, so it drops into CI unchanged.

## The write-up

The full debugging story behind both checks — how a black SDXL image led to a broken documented
contract, and how a five-month-old weight-corruption report was reproduced without downloading the
30 GB model:
**[Your Mac is quietly computing the wrong answer](docs/your-mac-is-computing-the-wrong-answer.md)**

## Why this exists

Both checks that ship in v0.1 came out of debugging real failures in
[🤗 diffusers](https://github.com/huggingface/diffusers), and each traced to a PyTorch backend
defect rather than to the library:

| Check | Real-world symptom | Upstream |
|---|---|---|
| `baddbmm(beta=0)` propagates NaN from an uninitialized `input` buffer | SDXL renders **all-black images** on Apple Silicon with attention slicing ([diffusers#14438](https://github.com/huggingface/diffusers/issues/14438)) | [pytorch#187521](https://github.com/pytorch/pytorch/issues/187521) — fixed in `main`, **not in v2.13.0** |
| `non_blocking` host→device copy reads source storage freed before the stream syncs | Model weights **silently corrupt** while loading with `device_map="mps"` ([diffusers#13227](https://github.com/huggingface/diffusers/issues/13227)) | [pytorch#189690](https://github.com/pytorch/pytorch/issues/189690) — open |

The pattern in both: PyTorch documents a guarantee, the MPS kernel does not honour it, and the
libraries built on top are written against the documented behaviour. Nothing in the stack notices.

## Observed on Apple M5 Pro / macOS 26.4.1

| torch | `baddbmm(beta=0)` | `non_blocking` copy |
|---|---|---|
| 2.10.0 | corrupt | corrupt |
| 2.11.0 | corrupt | corrupt |
| 2.12.0 | corrupt | corrupt |
| 2.13.0 | **corrupt** | clean |

`baddbmm` is fixed on PyTorch `main` (commit `e18b4e69b`, 17 Jun 2026) but the fix missed the 2.13
release branch, so the current stable release is still affected — expect 2.14. Note the family is
not uniformly broken: `addmm(beta=0)` honours the contract on every version tested.

## Usage

```console
mps-sentry                          # run everything
mps-sentry --list                   # show available checks
mps-sentry --only baddbmm-beta0     # run one check
```

```python
import mps_sentry

for result in mps_sentry.run():
    if result.failed:
        print(result.name, result.detail)
```

## Running this in CI

Hosted macOS runners (including GitHub's `macos-14`) report `torch.backends.mps.is_available()`
as `True` but fail on the first allocation with *"MPS backend out of memory (MPS allocated:
0 bytes)"* — the flag alone is not a reliable signal. `mps_sentry.mps_usable()` attempts a real
allocation and kernel launch, and the CLI exits **2** with an explanation rather than crashing
when no usable device is present. Exit codes:

| code | meaning |
|---|---|
| 0 | no silent-correctness failures found |
| 1 | at least one silent-correctness failure |
| 2 | no usable MPS device on this machine |

So a self-hosted Apple Silicon runner gates on `mps-sentry`, and a hosted runner can tolerate
exit 2 without pretending it verified anything.

## Reading the output

- **CORRUPT** — MPS and CPU disagree beyond what the dtype's rounding allows, or NaN appears only
  on MPS. This is a real correctness failure on your build.
- **ok** — agreement within tolerance (`5e-2` fp16, `2e-1` bf16, `1e-4` fp32), chosen so that
  normal low-precision accumulation differences are not reported as bugs.
- **skip** — not applicable to this backend (e.g. MPS has no float64, so hardcoded float64 raises
  rather than corrupting).

One honest caveat: the `non_blocking` failure is allocator-dependent and appears in only a few
tensors per batch, so the probe repeats the scenario several times. A clean result lowers the
probability that your build is affected — it is not proof of absence.

## Contributing a check

A check is a function returning a `ProbeResult`. Build a scenario, run it on MPS, compare with a CPU
reference over the same inputs, and hand both to `_verdict`. If you have hit a silent MPS
miscompute, a probe for it is very welcome — that is the point of the project.

## License

Apache-2.0
