# Your Mac is quietly computing the wrong answer

Two weeks ago someone filed a bug report against 🤗 diffusers: on Apple Silicon, Stable Diffusion XL
produced a perfectly black image. Twenty denoising steps, no error, no warning, an 843-byte PNG of
nothing.

I have an M5 Pro, so I picked it up. What I expected was a memory bug on 8 GB Macs, which is what the
report suggested. What I found, two days later, was a PyTorch kernel quietly breaking a promise its
own documentation makes — and then, chasing the same class of failure, a second bug that silently
corrupts model weights as they load.

Both are the interesting kind of bug: no crash, no warning, nothing in the logs. Just wrong numbers.

## Bug one: the buffer that was supposed to be ignored

The first useful move was to stop trusting where the symptom appeared. Image generation is a
pipeline — text encoder, then twenty rounds of denoising in a compressed latent space, then a VAE
that decodes latents into pixels. A bad number anywhere flows downstream, so the place you *see*
damage is rarely the place it starts.

The report blamed the VAE, the last stage. I put health checks between stages instead. The latents
arriving at the VAE were already entirely NaN — and the VAE's own weights were clean. It was
faithfully decoding garbage that was handed to it.

So I moved the checks into the denoising loop, and got the first real fact: **step 0 computed
correctly, then step 1 received clean inputs and emitted NaN.** Something inside a single UNet
forward pass was manufacturing NaN out of finite inputs.

Then the investigation got strange. I attached detectors to all ~2,000 submodules to find the exact
layer, and the bug vanished. Every run clean.

A bug that disappears when you observe it is not a mystery, it is a clue: my detectors were forcing
the GPU to finish each operation before the next began. If serialising the work hides the bug, the
bug lives in memory or timing, not arithmetic. So I stopped watching and started varying the
configuration instead — and a matrix over offloading modes and attention settings, three seeds each,
gave a clean answer: the trigger was `enable_attention_slicing()`, the memory-saving option that
computes attention in chunks.

Reading that code path led to one line in `get_attention_scores`:

```python
baddbmm_input = torch.empty(batch, tokens, tokens, dtype=query.dtype, device=query.device)
attention_scores = torch.baddbmm(baddbmm_input, query, key.transpose(-1, -2), beta=0, alpha=self.scale)
```

`baddbmm` computes `beta × input + alpha × (A @ B)`. Here `beta=0`, so the input contributes nothing
— which is exactly why the code can pass `torch.empty`, a buffer of uninitialised memory. It is only
there to receive the output shape. This is a deliberate, documented optimisation, and PyTorch's docs
are explicit about why it is safe:

> If `beta` is 0, then `input` will be ignored, and `nan` and `inf` in it will not be propagated.

That guarantee is load-bearing. Note it is a real promise, not a mathematical consequence — `0 × NaN`
is NaN, so the backend must *skip* the input rather than multiply it by zero. CPU and CUDA honour it.

MPS does not. Thirty lines, no diffusers involved:

```python
import torch

B, T, D = 10, 4096, 64
junk = torch.full((B, T, T), float("nan"), device="mps", dtype=torch.float16)
del junk                                     # freed — the allocator will recycle these pages
q = torch.randn(B, T, D, device="mps", dtype=torch.float16)
k = torch.randn(B, T, D, device="mps", dtype=torch.float16)
buf = torch.empty(B, T, T, device="mps", dtype=torch.float16)   # recycles the NaN pages

print(torch.isnan(torch.baddbmm(buf, q, k.mT, beta=0, alpha=0.125)).any())  # True  ← bug
print(torch.isnan(torch.bmm(q, k.mT) * 0.125).any())                        # False ← control
```

That is the whole bug. Uninitialised memory that happened to contain NaN bit patterns walked straight
through a parameter the documentation promises is ignored, into the attention scores, into the
latents, and out as a black image.

It also explains every strange thing about the symptom. Only the *sliced* attention processor calls
this code — the default path uses a fused kernel that never allocates such a buffer — so slicing
looked like the cause. Model offloading made it far more likely, because moving whole components in
and out of the GPU churns the allocator and leaves large freed regions for `torch.empty` to hand
back. And it was never about memory pressure: it reproduced identically on 24 GB.

The fix drops the buffer entirely, computing scores with a pre-scaled `bmm`. It relies on no contract,
allocates one fewer scores-sized tensor, and benchmarked ~37% faster than the code it replaced —
which is a pleasant outcome for a correctness patch.

**The part I got wrong.** In the pull request I said I would report this upstream. Then I searched
properly and found it already reported *and already fixed* — [pytorch#187521][1], patched in `main`
in June. I had to go back and correct my own claim publicly, which stung, but the correction turned
out to matter: the fix **missed the 2.13 release branch**. It is in `main`, not in any release. So the
current stable PyTorch still has it, and every Mac user on it is affected today.

## Bug two: weights that corrupt while you load them

The debugging pattern from the first bug — *stop trusting the symptom's location, then binary-search
the pipeline* — turned out to generalise. So I went looking for other silent failures on the same
hardware and found a five-month-old report with zero replies: loading a large model with
`device_map="mps"` silently corrupts some of its weights. Values around 1e37, LayerNorm overflow,
black images. Loading on CPU and then moving to the GPU works fine.

Five months of silence is usually a signal about difficulty, and here it was reproduction cost: the
model in the report is a 30 GB download. Rather than fetch it, I built a **faithful miniature** — a
three-layer transformer at the same architectural width, saved as an eleven-shard bf16 checkpoint,
about 2 GB generated locally in a minute. It reproduced the bug immediately, and — the detail that
made it credible — corrupted *the same named modules* the reporter's own instrumentation had flagged.

Reducing that to a raw-PyTorch probe gave the mechanism. The loader casts each tensor to the target
dtype, producing a short-lived CPU temporary, then enqueues an asynchronous copy to the GPU:

```python
value = value.to(dtype)                              # temporary
value.to("mps", non_blocking=True)                   # async copy; temporary dies immediately after
```

Across a sharded checkpoint that is hundreds of queued copies whose source tensors are freed while
the copies are still in flight. On MPS, `non_blocking=True` will happily read storage that has
already been released — [pytorch#189690][2], still open — so parameters silently take whatever now
occupies that memory.

Bisecting told the rest of the story:

| torch | corrupted tensors |
|---|---|
| 2.10.0 | 17 / 65 |
| 2.11.0 | 43 / 65 |
| 2.12.0 | 41 / 65 |
| 2.13.0 | 0 / 65 |
| any version, blocking copies | 0 / 65 |

Fixed in 2.13, broken in everything before it, and the fix is a guard that keeps the fast path
everywhere it is safe.

## What I actually learned

**Silent bugs get misattributed by default.** Both of these had been blamed on the wrong component —
the VAE in one case, the model checkpoint in the other — because the only visible evidence is at the
end of the pipeline. The first job is never to fix; it is to stop believing the symptom's address.

**A bug that hides when observed is telling you something.** Losing a day to a bug that vanished
under instrumentation felt like failure. It was the most informative result of the whole
investigation: it ruled out arithmetic and pointed at memory reuse.

**Verify the verifier.** Three separate times, my *checking* code had the bug rather than the code
under test — a stash that silently no-op'd, an argument that disabled the very feature I was
testing. Each time it was caught only because one number disagreed with five earlier runs. If a test
has never failed, you do not know it can.

**Documented guarantees are not guarantees.** Both bugs are the same shape at a different altitude: a
promise made in docs, unimplemented in one backend, relied upon by libraries above. This is a
structural weakness of a young accelerator backend, not two unlucky kernels.

## The tool

Both checks now live in **[mps-sentry][3]**, a small package that runs operations on MPS against a
CPU reference and reports only disagreement that float rounding cannot explain. It exits non-zero
when it finds silent miscompute, so it works in CI.

On my machine, torch 2.12 reports both failures and 2.13 reports one — the `baddbmm` contract
violation that never made the release. Useful nuance it also surfaced: `addmm(beta=0)` honours the
contract on every version I tested, so the family is not uniformly broken.

If you run models on a Mac and something is *slightly* wrong — worse outputs, occasional black
images, a quantised model that underperforms for no reason — it is worth thirty seconds to rule out
the backend before blaming your weights.

[1]: https://github.com/pytorch/pytorch/issues/187521
[2]: https://github.com/pytorch/pytorch/issues/189690
[3]: https://github.com/RudraMantri123/mps-sentry
