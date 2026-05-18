"""Benchmark CPU kernels in sgl-kernel vs PyTorch baseline.

Usage:
    python sgl-kernel/benchmark/bench_cpu_ops.py
    python sgl-kernel/benchmark/bench_cpu_ops.py --op rmsnorm
    python sgl-kernel/benchmark/bench_cpu_ops.py --op activation --num-iters 200
"""

import argparse
import itertools
import time

import torch
import torch.nn.functional as F

# Check CPU extension availability
try:
    torch.ops.load_library("sgl_kernel")
except Exception:
    pass

HAS_CPU_OPS = hasattr(torch.ops, "sgl_kernel") and hasattr(
    torch.ops.sgl_kernel, "silu_and_mul_cpu"
)


def bench_fn(fn, warmup=10, num_iters=100):
    """Benchmark a function, return median time in microseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(num_iters):
        start = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - start) / 1000)
    times.sort()
    return times[len(times) // 2]


# ============ Activation Benchmarks ============


def bench_activation(num_iters=100):
    print("\n=== Activation (silu_and_mul) ===")
    print(f"{'batch':>6} {'dim':>6} {'sgl(us)':>10} {'torch(us)':>10} {'speedup':>8}")

    batch_sizes = [1, 8, 32, 128, 512, 2048]
    dims = [2048, 7168, 14336, 27648]

    for bs, dim in itertools.product(batch_sizes, dims):
        x = torch.randn(bs, dim * 2, dtype=torch.bfloat16)

        def sgl_fn():
            return torch.ops.sgl_kernel.silu_and_mul_cpu(x)

        def torch_fn():
            a, b = x.chunk(2, dim=-1)
            return F.silu(a) * b

        t_sgl = bench_fn(sgl_fn, num_iters=num_iters)
        t_torch = bench_fn(torch_fn, num_iters=num_iters)
        speedup = t_torch / t_sgl
        print(f"{bs:>6} {dim:>6} {t_sgl:>10.1f} {t_torch:>10.1f} {speedup:>7.2f}x")


# ============ RMSNorm Benchmarks ============


def bench_rmsnorm(num_iters=100):
    print("\n=== RMSNorm ===")
    print(
        f"{'batch':>6} {'hidden':>6} {'sgl(us)':>10} {'torch(us)':>10} {'speedup':>8}"
    )

    batch_sizes = [1, 8, 32, 128, 512, 2048]
    hidden_sizes = [2048, 3584, 4096, 5120, 7168, 8192]
    eps = 1e-6

    for bs, hidden in itertools.product(batch_sizes, hidden_sizes):
        x = torch.randn(bs, hidden, dtype=torch.bfloat16)
        w = torch.randn(hidden, dtype=torch.bfloat16)

        def sgl_fn():
            return torch.ops.sgl_kernel.rmsnorm_cpu(x, w, eps)

        def torch_fn():
            variance = x.float().pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(variance + eps)) * w

        t_sgl = bench_fn(sgl_fn, num_iters=num_iters)
        t_torch = bench_fn(torch_fn, num_iters=num_iters)
        speedup = t_torch / t_sgl
        print(f"{bs:>6} {hidden:>6} {t_sgl:>10.1f} {t_torch:>10.1f} {speedup:>7.2f}x")


# ============ Fused Add RMSNorm Benchmarks ============


def bench_fused_add_rmsnorm(num_iters=100):
    print("\n=== Fused Add RMSNorm ===")
    print(
        f"{'batch':>6} {'hidden':>6} {'sgl(us)':>10} {'torch(us)':>10} {'speedup':>8}"
    )

    batch_sizes = [1, 8, 32, 128, 512]
    hidden_sizes = [2048, 3584, 4096, 8192]
    eps = 1e-6

    for bs, hidden in itertools.product(batch_sizes, hidden_sizes):
        w = torch.randn(hidden, dtype=torch.bfloat16)

        def sgl_fn():
            x = torch.randn(bs, hidden, dtype=torch.bfloat16)
            r = torch.randn(bs, hidden, dtype=torch.bfloat16)
            torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(x, r, w, eps)
            return x

        def torch_fn():
            x = torch.randn(bs, hidden, dtype=torch.bfloat16)
            r = torch.randn(bs, hidden, dtype=torch.bfloat16)
            x.add_(r)
            variance = x.float().pow(2).mean(-1, keepdim=True)
            x.copy_((x * torch.rsqrt(variance + eps)) * w)
            return x

        t_sgl = bench_fn(sgl_fn, num_iters=num_iters)
        t_torch = bench_fn(torch_fn, num_iters=num_iters)
        speedup = t_torch / t_sgl
        print(f"{bs:>6} {hidden:>6} {t_sgl:>10.1f} {t_torch:>10.1f} {speedup:>7.2f}x")


# ============ RoPE Benchmarks ============


def bench_rope(num_iters=100):
    print("\n=== RoPE (apply_rotary_pos_emb) ===")
    print(
        f"{'batch':>6} {'heads':>6} {'seq':>5} {'sgl(us)':>10} {'torch(us)':>10} {'speedup':>8}"
    )

    configs = [
        (1, 32, 1),  # decode
        (1, 32, 512),  # prefill
        (32, 32, 1),  # batched decode
        (32, 32, 512),  # batched prefill
    ]
    head_dim = 128

    for bs, num_heads, seq_len in configs:
        num_tokens = bs * seq_len
        # CPU RoPE expects 3D: [num_tokens, num_heads, head_dim]
        q = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16)
        k = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16)
        cos = torch.randn(num_tokens, head_dim, dtype=torch.float32)
        sin = torch.randn(num_tokens, head_dim, dtype=torch.float32)

        def sgl_fn():
            return torch.ops.sgl_kernel.apply_rotary_pos_emb_cpu(q, k, cos, sin)

        def torch_fn():
            # Standard RoPE: broadcast cos/sin over heads
            c = cos.unsqueeze(1).to(torch.bfloat16)
            s = sin.unsqueeze(1).to(torch.bfloat16)
            q_embed = (q * c) + (_rotate_half(q) * s)
            k_embed = (k * c) + (_rotate_half(k) * s)
            return q_embed, k_embed

        t_sgl = bench_fn(sgl_fn, num_iters=num_iters)
        t_torch = bench_fn(torch_fn, num_iters=num_iters)
        speedup = t_torch / t_sgl
        print(
            f"{bs:>6} {num_heads:>6} {seq_len:>5} {t_sgl:>10.1f} {t_torch:>10.1f} {speedup:>7.2f}x"
        )


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# ============ TopK Benchmarks ============


def bench_topk(num_iters=100):
    print("\n=== TopK Softmax (MoE gating) ===")
    print(
        f"{'tokens':>7} {'experts':>8} {'topk':>5} {'sgl(us)':>10} {'torch(us)':>10} {'speedup':>8}"
    )

    configs = [
        (32, 8, 2),  # small MoE
        (128, 64, 6),  # DeepSeek-style
        (512, 64, 6),
        (2048, 256, 8),  # large MoE
    ]

    for num_tokens, num_experts, topk in configs:
        hidden = torch.randn(num_tokens, 1024, dtype=torch.bfloat16)
        gating = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16)

        def sgl_fn():
            return torch.ops.sgl_kernel.topk_softmax_cpu(
                hidden, gating, topk, True
            )

        def torch_fn():
            scores = F.softmax(gating, dim=-1)
            topk_weights, topk_ids = torch.topk(scores, topk, dim=-1)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            return topk_weights, topk_ids

        t_sgl = bench_fn(sgl_fn, num_iters=num_iters)
        t_torch = bench_fn(torch_fn, num_iters=num_iters)
        speedup = t_torch / t_sgl
        print(
            f"{num_tokens:>7} {num_experts:>8} {topk:>5} {t_sgl:>10.1f} {t_torch:>10.1f} {speedup:>7.2f}x"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU kernels")
    parser.add_argument(
        "--op",
        choices=["activation", "rmsnorm", "fused_add_rmsnorm", "rope", "topk", "all"],
        default="all",
    )
    parser.add_argument("--num-iters", type=int, default=100)
    args = parser.parse_args()

    if not HAS_CPU_OPS:
        print("ERROR: sgl_kernel CPU ops not available.")
        print("Install with: pip install sgl-kernel[cpu] or build from source.")
        return

    print(f"Benchmarking CPU ops (num_iters={args.num_iters})")
    print(f"PyTorch: {torch.__version__}")
    print(f"Threads: {torch.get_num_threads()}")

    ops = {
        "activation": bench_activation,
        "rmsnorm": bench_rmsnorm,
        "fused_add_rmsnorm": bench_fused_add_rmsnorm,
        "rope": bench_rope,
        "topk": bench_topk,
    }

    if args.op == "all":
        for fn in ops.values():
            fn(num_iters=args.num_iters)
    else:
        ops[args.op](num_iters=args.num_iters)


if __name__ == "__main__":
    main()
