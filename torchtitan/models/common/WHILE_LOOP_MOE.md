# While-Loop Chunked Expert Processing for MoE

## Motivation

MoE expert computation involves data-dependent routing: each step produces
different per-expert token counts, and under EP the total token count per rank
also varies (determined by the all-to-all dispatch).

`grouped_mm` itself handles data-dependent per-expert offsets internally — the
offsets are values, not shapes, so they don't cause graph breaks. The graphability
challenge comes from the **surrounding control flow and tensor allocations** that
depend on `total_tokens`:

- Without EP, `total_tokens = bs * slen * top_k` is static. The while_loop
  with fixed `num_chunks` and `chunk_size` produces a fully static computation
  graph that `torch.compile` can capture as a single `fullgraph=True` trace.

- With EP, `total_tokens` varies per step, so the outer allocations
  (`padded_total`, `num_chunks`) remain dynamic. The while_loop still
  stabilizes the per-chunk computation, but full graphability requires
  static-capacity EP dispatch (e.g. budgeted padding or HybridEP non-blocking
  mode), which is orthogonal to this change.

## Approach

We add an optional chunked expert processing path inside `GroupedExperts._experts_forward()`.
When `while_loop_chunk_size` is set, expert-sorted tokens are:

1. **Padded** to a multiple of `chunk_size` (padding uses a sentinel expert ID
   so it doesn't inflate any real expert's token count)
2. **Processed in fixed-size chunks** — each chunk runs `_run_experts_grouped_mm`
   on exactly `chunk_size` rows, with padding outputs masked to zero
3. **Assembled** via `scatter` back into the full output tensor

Under `torch.compile`, chunks are iterated via `torch.while_loop` (verified
`fullgraph=True` with zero graph breaks). In eager mode, a Python `for`-loop is
used instead because `torch.while_loop`'s autograd backward
(`WhileLoopAutogradOp` → `while_loop_stack_output`) is incompatible with
activation checkpointing's `_CachingTorchDispatchMode`.

### Integration level

The chunked path lives inside `GroupedExperts._experts_forward()`, making it
transparent to the EP dispatch/combine hooks which fire as pre/post hooks on
`GroupedExperts.forward()`. This means it works with all EP backends (standard
all-to-all, DeepEP, HybridEP) without any changes to the dispatch layer.

## Key Design Decisions

### Zero-token early return

When `total_tokens == 0` (possible under EP when load balancing routes no tokens
to a rank's experts), we must still call the expert function rather than
returning early. An early `return x` would skip the EP combine post-hook,
causing an all-to-all deadlock where the other rank waits indefinitely.

### Padding sentinel

Padding positions use `expert_id = num_experts` (one past the last valid expert
index). `histc` with `bins = num_experts + 1` puts padding into its own bin,
keeping real expert counts accurate. The padding bin is dropped before passing
counts to `grouped_mm`.

### NaN masking

`grouped_mm` produces undefined values for rows beyond its last offset (the
padding rows). These are masked to zero with `torch.where(is_real, chunk_out, 0)`
to prevent NaN propagation through backward.

### Numerical precision

With production-scale weight initialization (`trunc_normal_` std=0.02) and
non-degenerate inputs, chunked vs non-chunked paths match to:
- **Forward**: exact (0 diff)
- **x.grad**: exact (0 diff)
- **Weight gradients**: max abs diff ~0.002 (from bf16 `grouped_mm` accumulation
  order in `dW = x^T @ grad_y` when an expert's tokens are split across chunks)

Measured over 200 seeds with DIM=64, HIDDEN=128, 4 experts.

## Usage

```bash
# Enable via CLI on any MoE model:
NGPU=2 MODULE=deepseek_v3 CONFIG=deepseek_v3_16b_4layer ./run_train.sh \
    --parallelism.while_loop_chunk_size=4096

# Or set in a config registry function:
ParallelismConfig(
    expert_parallel_degree=2,
    while_loop_chunk_size=4096,
)
```

The CLI option works for all MoE models (deepseek_v3, llama4, gpt_oss) via the
`update_from_config` override in each model.

## Files Changed

| File | Change |
|------|--------|
| `torchtitan/models/common/moe.py` | `_run_experts_chunked_while_loop()`, `GroupedExperts.Config.while_loop_chunk_size` |
| `torchtitan/models/common/config_utils.py` | `while_loop_chunk_size` in `make_experts_config()` |
| `torchtitan/config/configs.py` | `ParallelismConfig.while_loop_chunk_size` CLI field |
| `torchtitan/models/deepseek_v3/__init__.py` | `_build_dsv3_layers` passthrough, `_16b_4layer` debug config |
| `torchtitan/models/deepseek_v3/config_registry.py` | `deepseek_v3_16b_4layer` training config |
| `torchtitan/models/deepseek_v3/model.py` | `update_from_config` override wiring |
| `torchtitan/models/llama4/model.py` | `update_from_config` override wiring |
| `torchtitan/models/gpt_oss/model.py` | `update_from_config` override wiring |
| `tests/unit_tests/test_while_loop_moe.py` | 7 tests (forward, backward, edge cases) |

## Testing

```bash
# Unit tests
pytest tests/unit_tests/test_while_loop_moe.py -v

# torch.compile fullgraph verification
python -c "
import torch, torch.nn as nn
from torchtitan.models.common.moe import _run_experts_chunked_while_loop
# ... (see test file for setup)
torch.compile(_run_experts_chunked_while_loop, fullgraph=True)(...)
"

# E2E training (2 GPU, EP=2)
NGPU=2 MODULE=deepseek_v3 CONFIG=deepseek_v3_16b_4layer ./run_train.sh \
    --parallelism.while_loop_chunk_size=4096 --training.steps=20
```
