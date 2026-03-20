"""
Benchmark script for SwiGLU FFN epilogue fusion with Triton GEMM templates.

Compares three configurations at MRS-relevant shapes:
1. Baseline: cuBLAS (ATEN backend)
2. Triton GEMM only (no epilogue fusion)
3. Triton GEMM + epilogue fusion

Reports latency, peak memory, and runtime kernel launch count for each
configuration.

NOTE: The real performance benefit of Triton GEMM + epilogue fusion is at the
full-model level (46 matmuls, ~36 eliminated pointwise kernels, ~200MB memory
savings). A standalone SwiGLU FFN (3 matmuls) may not show dramatic improvement
since cuBLAS is highly optimized for large shapes.

Usage:
    buck2 run @fbcode//mode/opt -c fbcode.enable_gpu_sections=true \
        fbcode//caffe2/test/inductor:run_benchmark_swiglu_fusion
"""

import argparse
import math
import time

import torch
import torch._inductor.config as inductor_config


class SwiGLUFFN(torch.nn.Module):
    """SwiGLU FFN matching the MRS model pattern: SiLU(x @ W_gate) * (x @ W_up) @ W_down"""

    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.w_gate = torch.nn.Linear(input_dim, hidden_dim, bias=bias)
        self.w_up = torch.nn.Linear(input_dim, hidden_dim, bias=bias)
        self.w_down = torch.nn.Linear(hidden_dim, input_dim, bias=bias)
        self.silu = torch.nn.SiLU()
        self._init_weights()

    def _init_weights(self) -> None:
        for name in ("w_gate", "w_up", "w_down"):
            layer = getattr(self, name)
            torch.nn.init.normal_(
                layer.weight, mean=0.0, std=1.0 / math.sqrt(layer.in_features)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.silu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        return self.w_down(hidden)


class SwiGLUFFNWithRMSNormResidual(torch.nn.Module):
    """Full MRS pattern: RMSNorm -> SwiGLU FFN -> residual add."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = torch.nn.RMSNorm(input_dim)
        self.ffn = SwiGLUFFN(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


def _do_bench(fn, warmup=1000, rep=2000):
    """Simple benchmark: run fn with warmup then measure rep iterations."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / rep * 1000  # ms


def _count_runtime_kernels(fn):
    """Count actual CUDA kernel launches using torch.cuda.Event profiling."""
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        fn()
        torch.cuda.synchronize()

    kernel_count = 0
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.count > 0:
            kernel_count += evt.count
    return kernel_count


def _benchmark_config(
    model_cls, input_shape, dtype, config_dict, config_name, include_backward
):
    """Compile model with given config and benchmark it."""
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    input_dim = input_shape[1]
    hidden_dim = input_dim * 2

    model = model_cls(input_dim, hidden_dim).to(device="cuda", dtype=dtype)
    x = torch.randn(*input_shape, device="cuda", dtype=dtype)

    with inductor_config.patch(config_dict):
        compiled_model = torch.compile(model, fullgraph=True)

        if include_backward:
            x_grad = x.detach().clone().requires_grad_(True)

            def fn():
                out = compiled_model(x_grad)
                out.sum().backward(retain_graph=True)
        else:

            def fn():
                compiled_model(x)

        # Warmup (includes compilation)
        for _ in range(3):
            fn()
        torch.cuda.synchronize()

    # Count actual runtime kernel launches
    kernel_count = _count_runtime_kernels(fn)

    # Measure peak memory
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Benchmark latency
    latency_ms = _do_bench(fn)

    return {
        "config": config_name,
        "kernel_launches": kernel_count,
        "latency_ms": latency_ms,
        "peak_memory_mb": peak_mem_mb,
    }


CONFIGS = {
    "cuBLAS (ATEN)": {
        "max_autotune": False,
        "epilogue_fusion": False,
    },
    "Triton GEMM (no fusion)": {
        "max_autotune": True,
        "max_autotune_gemm_backends": "TRITON",
        "epilogue_fusion": False,
        "autotune_fallback_to_aten": False,
    },
    "Triton GEMM + epilogue fusion": {
        "max_autotune": True,
        "max_autotune_gemm_backends": "TRITON",
        "epilogue_fusion": True,
        "autotune_fallback_to_aten": False,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark SwiGLU FFN epilogue fusion")
    parser.add_argument(
        "--batch-dim",
        type=int,
        default=5120,
        help="Batch dimension (default: 5120, matching MRS Region [11/0])",
    )
    parser.add_argument(
        "--input-dim", type=int, default=4096, help="Input dimension (default: 4096)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type (default: bf16)",
    )
    parser.add_argument(
        "--include-backward",
        action="store_true",
        help="Include backward pass in benchmark",
    )
    parser.add_argument(
        "--with-rmsnorm",
        action="store_true",
        help="Include RMSNorm + residual (full MRS pattern)",
    )
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    input_shape = (args.batch_dim, args.input_dim)
    model_cls = SwiGLUFFNWithRMSNormResidual if args.with_rmsnorm else SwiGLUFFN

    model_name = model_cls.__name__
    print(f"\n{'=' * 72}")
    print("SwiGLU FFN Epilogue Fusion Benchmark")
    print(f"{'=' * 72}")
    print(f"Model:      {model_name}")
    print(
        f"Shape:      [{args.batch_dim}, {args.input_dim}]"
        f" -> hidden [{args.batch_dim}, {args.input_dim * 2}]"
    )
    print(f"Dtype:      {args.dtype}")
    print(f"Backward:   {args.include_backward}")
    print(f"{'=' * 72}\n")

    results = []
    for config_name, config_dict in CONFIGS.items():
        print(f"Benchmarking: {config_name}...")
        result = _benchmark_config(
            model_cls=model_cls,
            input_shape=input_shape,
            dtype=dtype,
            config_dict=config_dict,
            config_name=config_name,
            include_backward=args.include_backward,
        )
        results.append(result)
        print(
            f"  Kernel launches: {result['kernel_launches']:>3}  "
            f"Latency: {result['latency_ms']:>8.3f} ms  "
            f"Peak Mem: {result['peak_memory_mb']:>8.1f} MB"
        )

    # Summary table
    print(f"\n{'=' * 72}")
    print(
        f"{'Configuration':<35} {'Launches':>8} {'Latency(ms)':>12}"
        f" {'Peak Mem(MB)':>13}"
    )
    print(f"{'-' * 72}")
    for r in results:
        print(
            f"{r['config']:<35} {r['kernel_launches']:>8}"
            f" {r['latency_ms']:>12.3f} {r['peak_memory_mb']:>13.1f}"
        )
    print(f"{'=' * 72}")

    # Compute speedups relative to cuBLAS baseline
    if len(results) >= 2:
        baseline = results[0]
        print(f"\nComparison vs {baseline['config']}:")
        for r in results[1:]:
            speedup = baseline["latency_ms"] / r["latency_ms"]
            kernel_diff = r["kernel_launches"] - baseline["kernel_launches"]
            mem_saving = baseline["peak_memory_mb"] - r["peak_memory_mb"]
            print(
                f"  {r['config']}: "
                f"{speedup:.2f}x latency, "
                f"{kernel_diff:+d} kernel launches, "
                f"{mem_saving:+.1f} MB memory"
            )

    print(
        "\nNOTE: Full-model benefit (46 matmuls) is measured via"
        " rllayer_benchmark --triton-gemm"
    )


if __name__ == "__main__":
    main()
