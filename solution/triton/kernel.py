"""
Fused MoE Kernel for FlashInfer MLSys 2026 Contest.

Track: fused_moe (FP8 block-scale, DeepSeek-V3 routing)
Author: DiegoCao

Architecture:
  1. DeepSeek-V3 routing (PyTorch)
  2. Token permutation → sorted grouped layout
  3. GEMM1 (Triton): FP8 A × FP8 B with block-scale dequant → [T*8, 4096]
  4. SwiGLU (Triton): silu(up) * gate → [T*8, 2048]
  5. GEMM2 (Triton): FP32 A × FP8 B with block-scale dequant + routing weight → output
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Constants (DeepSeek-V3/R1 geometry)
# ============================================================================
H = 7168
I = 2048
GEMM1_OUT = 4096          # 2 * I
E_GLOBAL = 256
E_LOCAL = 32
BLOCK = 128               # FP8 quant block size
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
GROUP_SIZE_E = E_GLOBAL // N_GROUP  # 32 experts per group


# ============================================================================
# Triton: Grouped GEMM Kernel with FP8 block-scale dequant (for GEMM1)
# ============================================================================
@triton.jit
def grouped_gemm_fp8_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    sorted_slot_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    # Dimensions
    N, K, EM, num_valid_slots,
    # A strides [num_tokens, K]
    stride_am, stride_ak,
    # B strides [E_local, N, K]
    stride_be, stride_bn, stride_bk,
    # C strides [EM, N]
    stride_cm, stride_cn,
    # A_scale strides: [K//128, num_tokens]
    stride_as_block, stride_as_token,
    # B_scale strides: [E_local, N//128, K//128]
    stride_bs_e, stride_bs_n, stride_bs_k,
    # Meta
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_slot = tl.load(sorted_slot_ids_ptr + offs_m)
    token_mask = offs_slot < num_valid_slots

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert < 0:
        return

    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    original_tokens = offs_slot // top_k
    a_ptrs = a_ptr + (original_tokens[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = (
        b_ptr + off_expert * stride_be
        + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
    )

    a_scale_base = a_scale_ptr + original_tokens * stride_as_token
    offs_bs_n = offs_n // BLOCK_SIZE_K  # N block index (BLOCK_SIZE_K == quant block == 128)
    b_scale_base = b_scale_ptr + off_expert * stride_bs_e + offs_bs_n * stride_bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        a_scale = tl.load(a_scale_base + k * stride_as_block, mask=token_mask, other=0.0)
        b_scale = tl.load(b_scale_base + k * stride_bs_k)

        accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ============================================================================
# Triton: GEMM2 Kernel (FP32 input × FP8 weight with block-scale dequant)
# + routing weight multiplication
# ============================================================================
@triton.jit
def grouped_gemm2_kernel(
    # Pointers
    a_ptr,          # intermediate2: [EM, I] float32
    b_ptr,          # gemm2_weights: [E_local, H, I] fp8
    c_ptr,          # output_fp32: [T, H] float32 (fused reduce target)
    b_scale_ptr,    # gemm2_weights_scale: [E_local, H//128, I//128]
    topk_weights_ptr,  # sorted_topk_weights: [EM] float32
    sorted_slot_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    # Dimensions
    N, K, EM, num_valid_slots,
    # A strides [EM, K=I=2048]
    stride_am, stride_ak,
    # B strides [E_local, N=H=7168, K=I=2048]
    stride_be, stride_bn, stride_bk,
    # C strides [T, N=H=7168]
    stride_cm, stride_cn,
    # B_scale strides [E_local, N//128, K//128]
    stride_bs_e, stride_bs_n, stride_bs_k,
    # Quant block size
    QUANT_BLOCK: tl.constexpr,
    # Meta
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    compute_type: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
):
    """
    GEMM2 with fused reduce: computes A[slot] @ B[expert].T * weight,
    then atomic-adds directly to output[token, :] (slot//top_k → token).
    Eliminates the separate reduce kernel and intermediate3 buffer.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_slot = tl.load(sorted_slot_ids_ptr + offs_m)
    token_mask = offs_slot < num_valid_slots

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert < 0:
        return

    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A is already packed in sorted local-slot order.
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B: [E, N, K] loaded as [K, N] tiles
    b_ptrs = (
        b_ptr + off_expert * stride_be
        + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
    )

    # B scale pointers
    offs_bs_n = offs_n // QUANT_BLOCK
    b_scale_base = b_scale_ptr + off_expert * stride_bs_e + offs_bs_n * stride_bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        # Only B needs dequant (A is already float32)
        b_scale = tl.load(b_scale_base + k * stride_bs_k)  # [BLOCK_SIZE_N]

        # a is float32 [M, K], b is fp8 [K, N]
        # dot(float32, fp8) — cast b to float32 for accumulation
        accumulator += tl.dot(a, b.to(tl.float32)) * b_scale[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply routing weights
    if MUL_ROUTED_WEIGHT:
        w = tl.load(topk_weights_ptr + offs_m, mask=token_mask, other=0.0)
        accumulator = accumulator * w[:, None]

    # Fused reduce: atomic-add to output[original_token, :] (slot // top_k)
    original_tokens = (offs_slot // top_k).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + original_tokens[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")


# ============================================================================
# Triton: SwiGLU kernel
# ============================================================================
@triton.jit
def swiglu_kernel(
    input_ptr, output_ptr,
    M, half_N,
    stride_in_m, stride_in_n,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """SwiGLU: output = silu(x2) * x1, where input = [x1 | x2]"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < half_N)

    x1_ptrs = input_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n
    x2_ptrs = input_ptr + offs_m[:, None] * stride_in_m + (offs_n[None, :] + half_N) * stride_in_n

    x1 = tl.load(x1_ptrs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptrs, mask=mask, other=0.0).to(tl.float32)
    result = (x2 * tl.sigmoid(x2)) * x1

    out_ptrs = output_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, result, mask=mask)


# ============================================================================
# Triton: Reduce kernel — accumulate per-slot results into per-token output
# ============================================================================
@triton.jit
def reduce_topk_kernel(
    src_ptr, dst_ptr,
    T, N, top_k: tl.constexpr,
    stride_src_m, stride_src_n,
    stride_dst_m, stride_dst_n,
    BLOCK_N: tl.constexpr,
):
    """Reduce: dst[t, n] = sum_{k=0..top_k-1} src[t*top_k + k, n]"""
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_t >= T:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in range(top_k):
        slot = pid_t * top_k + k
        ptrs = src_ptr + slot * stride_src_m + offs_n * stride_src_n
        val = tl.load(ptrs, mask=mask_n, other=0.0)
        acc += val

    dst_ptrs = dst_ptr + pid_t * stride_dst_m + offs_n * stride_dst_n
    tl.store(dst_ptrs, acc.to(tl.bfloat16), mask=mask_n)


# ============================================================================
# Token permutation (pure PyTorch)
# ============================================================================
def build_local_expert_layout(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    block_size: int,
    local_expert_offset: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack only local-expert slots to avoid launching empty expert blocks."""
    T, top_k = topk_ids.shape
    device = topk_ids.device
    total_slots = T * top_k

    flat_ids = (
        torch.arange(T, device=device, dtype=torch.int32).unsqueeze(1) * top_k
        + torch.arange(top_k, device=device, dtype=torch.int32).unsqueeze(0)
    ).reshape(-1)
    flat_experts = topk_ids.reshape(-1).to(torch.int32)
    flat_weights = topk_weights.reshape(-1).float()

    local_end = local_expert_offset + num_local_experts
    local_mask = (flat_experts >= local_expert_offset) & (flat_experts < local_end)
    local_slot_ids = flat_ids[local_mask]
    local_expert_ids = (flat_experts[local_mask] - local_expert_offset).to(torch.int32)
    local_weights = flat_weights[local_mask]
    num_local_slots = int(local_slot_ids.numel())

    if num_local_slots == 0:
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        empty_f32 = torch.empty((0,), dtype=torch.float32, device=device)
        num_tokens_post_padded = torch.tensor([0], dtype=torch.int32, device=device)
        return empty_i32, empty_i32, empty_f32, num_tokens_post_padded, 0

    # Sort by local expert (stable to preserve token order within expert).
    sorted_expert_vals, sort_indices = local_expert_ids.sort(stable=True)
    sorted_slot_ids = local_slot_ids[sort_indices]
    sorted_weights = local_weights[sort_indices]

    expert_counts = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
    unique_experts, counts = sorted_expert_vals.unique_consecutive(return_counts=True)
    expert_counts[unique_experts.long()] = counts.int()

    expert_counts_cpu = expert_counts.cpu()
    padded_counts_cpu = ((expert_counts_cpu + block_size - 1) // block_size) * block_size
    num_blocks_cpu = padded_counts_cpu // block_size
    padded_offsets_cpu = torch.zeros(num_local_experts + 1, dtype=torch.int32)
    torch.cumsum(padded_counts_cpu, dim=0, out=padded_offsets_cpu[1:])
    src_offsets_cpu = torch.zeros(num_local_experts + 1, dtype=torch.int32)
    torch.cumsum(expert_counts_cpu, dim=0, out=src_offsets_cpu[1:])

    total_padded = padded_offsets_cpu[-1].item()
    sorted_slot_ids_out = torch.full((total_padded,), total_slots, dtype=torch.int32, device=device)
    sorted_weights_out = torch.zeros((total_padded,), dtype=torch.float32, device=device)
    max_blocks = (total_padded + block_size - 1) // block_size
    expert_ids_out = torch.full((max_blocks,), -1, dtype=torch.int32, device=device)

    active_mask = expert_counts_cpu > 0
    active_experts = torch.where(active_mask)[0]

    for e_idx in active_experts.tolist():
        cnt = expert_counts_cpu[e_idx].item()
        src_off = src_offsets_cpu[e_idx].item()
        dst_off = padded_offsets_cpu[e_idx].item()
        sorted_slot_ids_out[dst_off:dst_off + cnt] = sorted_slot_ids[src_off:src_off + cnt]
        sorted_weights_out[dst_off:dst_off + cnt] = sorted_weights[src_off:src_off + cnt]

        n_blk = num_blocks_cpu[e_idx].item()
        blk_start = dst_off // block_size
        expert_ids_out[blk_start:blk_start + n_blk] = e_idx

    num_tokens_post_padded = torch.tensor([total_padded], dtype=torch.int32, device=device)
    return (
        sorted_slot_ids_out,
        expert_ids_out,
        sorted_weights_out,
        num_tokens_post_padded,
        num_local_slots,
    )


# ============================================================================
# Entry point
# ============================================================================
@torch.no_grad()
def kernel(
    routing_logits: torch.Tensor,        # [T, 256] float32
    routing_bias: torch.Tensor,          # [256] bfloat16
    hidden_states: torch.Tensor,         # [T, 7168] float8_e4m3fn
    hidden_states_scale: torch.Tensor,   # [56, T] float32
    gemm1_weights: torch.Tensor,         # [32, 4096, 7168] float8_e4m3fn
    gemm1_weights_scale: torch.Tensor,   # [32, 32, 56] float32
    gemm2_weights: torch.Tensor,         # [32, 7168, 2048] float8_e4m3fn
    gemm2_weights_scale: torch.Tensor,   # [32, 56, 16] float32
    local_expert_offset: int,
    routed_scaling_factor: float,
    output: torch.Tensor,                # [T, 7168] bfloat16 (DPS)
):
    T = routing_logits.shape[0]
    device = routing_logits.device
    local_start = int(local_expert_offset)

    # ── 1. DeepSeek-V3 Routing ─────────────────────────────────────────────
    logits = routing_logits.float()
    bias = routing_bias.float().reshape(-1)

    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    s_grouped = s_with_bias.view(T, N_GROUP, GROUP_SIZE_E)
    top2_vals, _ = torch.topk(s_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros(T, N_GROUP, device=device)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, GROUP_SIZE_E).reshape(T, E_GLOBAL)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    weight_mask = torch.zeros_like(s)
    weight_mask.scatter_(1, topk_idx, 1.0)
    weights = s * weight_mask
    weights = (weights / (weights.sum(dim=1, keepdim=True) + 1e-20)) * routed_scaling_factor

    topk_weights = torch.gather(weights, 1, topk_idx)  # [T, 8]

    # ── 2. Token permutation ───────────────────────────────────────────────
    num_slots = T * TOP_K

    # Seq_len-aware BLOCK_M selection (reduces padding waste for small T)
    if T <= 8:
        BLOCK_M = 16
        GROUP_SIZE_M = 1
    elif T <= 32:
        BLOCK_M = 32
        GROUP_SIZE_M = 4
    else:
        BLOCK_M = 64
        GROUP_SIZE_M = 8

    (
        sorted_slot_ids,
        expert_ids_local,
        sorted_topk_weights,
        num_tokens_post_padded,
        num_local_slots,
    ) = build_local_expert_layout(
        topk_idx.int(), topk_weights, BLOCK_M, local_start, E_LOCAL,
    )

    if num_local_slots == 0:
        output.zero_()
        return output

    EM = sorted_slot_ids.shape[0]
    BLOCK_N = 128
    BLOCK_K = 128

    # ── 3. GEMM1: Triton grouped GEMM with FP8 dequant ────────────────────
    intermediate1 = torch.zeros((EM, GEMM1_OUT), dtype=torch.float32, device=device)

    grid1 = (triton.cdiv(EM, BLOCK_M) * triton.cdiv(GEMM1_OUT, BLOCK_N),)
    grouped_gemm_fp8_kernel[grid1](
        hidden_states, gemm1_weights, intermediate1,
        hidden_states_scale, gemm1_weights_scale,
        sorted_slot_ids, expert_ids_local, num_tokens_post_padded,
        GEMM1_OUT, H, EM, num_slots,
        hidden_states.stride(0), hidden_states.stride(1),
        gemm1_weights.stride(0), gemm1_weights.stride(1), gemm1_weights.stride(2),
        intermediate1.stride(0), intermediate1.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M, top_k=TOP_K, compute_type=tl.float32,
    )

    # ── 4. SwiGLU ──────────────────────────────────────────────────────────
    intermediate2 = torch.empty((EM, I), dtype=torch.float32, device=device)
    SWIGLU_BM = BLOCK_M
    SWIGLU_BN = 64
    swiglu_grid = (triton.cdiv(EM, SWIGLU_BM), triton.cdiv(I, SWIGLU_BN))
    swiglu_kernel[swiglu_grid](
        intermediate1, intermediate2,
        EM, I,
        intermediate1.stride(0), intermediate1.stride(1),
        intermediate2.stride(0), intermediate2.stride(1),
        BLOCK_M=SWIGLU_BM, BLOCK_N=SWIGLU_BN,
    )

    # ── 5. GEMM2 + Fused Reduce: atomic-add directly to per-token output ──
    # Eliminates intermediate3 buffer and separate reduce kernel
    output_fp32 = torch.zeros((T, H), dtype=torch.float32, device=device)

    grid2 = (triton.cdiv(EM, BLOCK_M) * triton.cdiv(H, BLOCK_N),)
    grouped_gemm2_kernel[grid2](
        intermediate2, gemm2_weights, output_fp32,
        gemm2_weights_scale,
        sorted_topk_weights,
        sorted_slot_ids, expert_ids_local, num_tokens_post_padded,
        H, I, EM, num_slots,
        intermediate2.stride(0), intermediate2.stride(1),
        gemm2_weights.stride(0), gemm2_weights.stride(1), gemm2_weights.stride(2),
        output_fp32.stride(0), output_fp32.stride(1),
        gemm2_weights_scale.stride(0), gemm2_weights_scale.stride(1), gemm2_weights_scale.stride(2),
        QUANT_BLOCK=BLOCK,
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M, compute_type=tl.float32, MUL_ROUTED_WEIGHT=True,
        top_k=TOP_K,
    )

    # Convert FP32 accumulation buffer to BF16 output
    output.copy_(output_fp32.to(torch.bfloat16))

    return output
