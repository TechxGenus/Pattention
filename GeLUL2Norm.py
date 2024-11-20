import torch
import math

import triton
import triton.language as tl

@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def _gelu_l2_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    Norm,  # pointer to the L2 norm
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    sqrt_N,  # sqrt of N
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    
    # Compute L2 norm
    _norm = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        _norm += x * x
    
    # Sum and sqrt
    l2_norm = tl.sqrt(tl.sum(_norm, axis=0) + eps)
    
    # Store the norm
    tl.store(Norm + row, l2_norm)
    
    # Normalize, multiply by sqrt(N), and apply GELU
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        y = (x / l2_norm) * sqrt_N
        # Apply GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        y = y * 0.5 * (1 + tanh(0.797885 * y + 0.035677 * y * y * y))
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def _gelu_l2_norm_bwd_fused(
    DX, DY, X, Norm,
    stride, N, sqrt_N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    
    # Offset pointers
    X += row * stride
    DY += row * stride
    DX += row * stride
    
    # Load norm and compute intermediate values
    norm = tl.load(Norm + row)
    norm_cube = norm * norm * norm
    dot = tl.zeros([1], dtype=tl.float32)
    
    # First pass: compute normalized values and GELU gradients    
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        normalized = (x / norm) * sqrt_N
        
        # GELU gradient computation
        tanh_arg = 0.797885 * normalized + 0.035677 * normalized * normalized * normalized
        tanh_val = tanh(tanh_arg)
        gelu_grad = 0.5 * (1 + tanh_val) + normalized * 0.5 * (1 - tanh_val * tanh_val) * (0.797885 + 0.107031 * normalized * normalized)
        
        dy = tl.load(DY + cols, mask=mask, other=0.).to(tl.float32)
        dot += tl.sum(x * (dy * gelu_grad), axis=0)
    
    # Second pass: compute final gradients
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        normalized = (x / norm) * sqrt_N
        
        # Recompute GELU gradient
        tanh_arg = 0.797885 * normalized + 0.035677 * normalized * normalized * normalized
        tanh_val = tanh(tanh_arg)
        gelu_grad = 0.5 * (1 + tanh_val) + normalized * 0.5 * (1 - tanh_val * tanh_val) * (0.797885 + 0.107031 * normalized * normalized)
        
        dy = tl.load(DY + cols, mask=mask, other=0.).to(tl.float32)
        dx = (dy * gelu_grad / norm) - (x * dot / norm_cube)
        dx = dx * sqrt_N
        tl.store(DX + cols, dx, mask=mask)

class GeLUL2Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps=1e-6):
        # Allocate output
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        norm = torch.empty((M,), dtype=torch.float32, device=x.device)
        
        # Setting the block size and number of warps
        BLOCK_SIZE = 1024
        num_warps = 4
        
        # Save for backward
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        
        sqrt_N = math.sqrt(float(N))
        
        # Launch kernel
        _gelu_l2_norm_fwd_fused[(M,)](
            x_arg, y, norm,
            x_arg.stride(0), N, eps, sqrt_N,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
        
        ctx.save_for_backward(x, norm)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, norm = ctx.saved_tensors
        dx = torch.empty_like(dy)
        
        # Launch backward kernel
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        
        sqrt_N = math.sqrt(float(N))
        
        _gelu_l2_norm_bwd_fused[(M,)](
            dx, dy, x, norm,
            x_arg.stride(0), N, sqrt_N,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps
        )
        return dx, None

def test_gelu_l2_norm(M, N, dtype, eps=1e-6, device='cuda'):
    # create data
    x_shape = (M, N)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    
    # forward pass
    y_tri = GeLUL2Norm.apply(x, eps)
    y_ref = torch.nn.functional.gelu(torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps).to(dtype) * math.sqrt(N))
    
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri = x.grad.clone()
    x.grad = None
    
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref = x.grad.clone()
    
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch', 'torch_compiled'],
        line_names=['Triton', 'Torch', 'Torch Compiled'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='l2-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'},
    ))
def bench_l2_norm(M, N, dtype, provider, mode='backward', eps=1e-6, device='cuda'):
    # create data
    x_shape = (M, N)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def torch_fn(x):
        return torch.nn.functional.gelu(torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps) * math.sqrt(N))

    compiled_fn = torch.compile(torch_fn)

    def y_fwd():
        if provider == "triton":
            return GeLUL2Norm.apply(x, eps)
        if provider == "torch":
            return torch_fn(x)
        if provider == "torch_compiled":
            return compiled_fn(x)

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

# Update test calls
test_gelu_l2_norm(1151, 8192, torch.float16)
bench_l2_norm.run(save_path='.', print_data=True)
