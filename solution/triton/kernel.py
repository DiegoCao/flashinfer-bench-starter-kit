"""
Sparse Attention Kernel (DSA) for FlashInfer MLSys 2026 Contest.

Track: sparse_attention (DeepSeek-V3.2 Native Sparse Attention)
Definition: dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
Author: DiegoCao

Architecture (MLA Sparse Attention):
  - Query: q_nope [num_tokens, 16, 512] + q_pe [num_tokens, 16, 64]
  - KV cache: ckv_cache [num_pages, 64, 512] + kpe_cache [num_pages, 64, 64]
  - Sparse indices select top-2048 KV entries per token
  - Attention: logits = (q_nope @ Kc.T + q_pe @ Kp.T) * sm_scale
  - Output: softmax(logits) @ Kc
"""

import math
import torch
import triton
import triton.language as tl


# ============================================================================
# Triton: Sparse Attention Kernel (d-tile decomposition)
# ============================================================================
# Each program handles one (token, head, d_tile) triple.
# - Computes full logits over all top-K KV entries (tiles over CKV + KPE dims)
# - Uses online softmax across KV chunks
# - Accumulates only its d_tile slice of the output
# - K is read once for logits, once for output per KV chunk (stays in L2)

@triton.jit
def _sparse_attn_dtile_kernel(
    # Pointers
    q_nope_ptr, q_pe_ptr,
    ckv_flat_ptr, kpe_flat_ptr,
    sparse_indices_ptr,
    output_ptr, lse_ptr,
    # Scalar
    sm_scale,
    # Dimensions
    num_tokens,
    topk,
    # Q strides (contiguous: [num_tokens, 16, dim])
    stride_qn_t, stride_qn_h,
    stride_qp_t, stride_qp_h,
    # Output strides
    stride_o_t, stride_o_h,
    stride_lse_t, stride_lse_h,
    # Constexprs
    BLOCK_KV: tl.constexpr,
    BLOCK_D_CKV: tl.constexpr,
    BLOCK_D_KPE: tl.constexpr,
    NUM_D_TILES: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    TOPK: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    d_tile_id = tl.program_id(2)

    d_out_start = d_tile_id * BLOCK_D_CKV

    # Sparse indices base for this token
    si_base = sparse_indices_ptr + token_id * TOPK

    # Online softmax state
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D_CKV], dtype=tl.float32)

    for kv_start in range(0, TOPK, BLOCK_KV):
        kv_offs = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offs < TOPK

        indices = tl.load(si_base + kv_offs, mask=kv_mask, other=-1)
        valid_mask = (indices != -1) & kv_mask
        # Cast indices to int64 for address computation
        indices_i64 = indices.to(tl.int64)

        # ---- Compute logits [BLOCK_KV] ----
        logits = tl.zeros([BLOCK_KV], dtype=tl.float32)

        # CKV contribution: dot(q_nope, Kc) tiled over head_dim
        for d_start in tl.static_range(0, HEAD_DIM_CKV, BLOCK_D_CKV):
            d_offs = d_start + tl.arange(0, BLOCK_D_CKV)
            q_chunk = tl.load(
                q_nope_ptr + token_id * stride_qn_t + head_id * stride_qn_h + d_offs
            ).to(tl.float32)
            k_addrs = indices_i64[:, None] * HEAD_DIM_CKV + d_offs[None, :]
            k_chunk = tl.load(
                ckv_flat_ptr + k_addrs,
                mask=valid_mask[:, None], other=0.0
            ).to(tl.float32)
            logits += tl.sum(q_chunk[None, :] * k_chunk, axis=1)

        # KPE contribution: dot(q_pe, Kp)
        d_offs_kpe = tl.arange(0, BLOCK_D_KPE)
        kpe_dim_mask = d_offs_kpe < HEAD_DIM_KPE
        q_pe_chunk = tl.load(
            q_pe_ptr + token_id * stride_qp_t + head_id * stride_qp_h + d_offs_kpe,
            mask=kpe_dim_mask, other=0.0
        ).to(tl.float32)
        kp_addrs = indices_i64[:, None] * HEAD_DIM_KPE + d_offs_kpe[None, :]
        kp_chunk = tl.load(
            kpe_flat_ptr + kp_addrs,
            mask=valid_mask[:, None] & kpe_dim_mask[None, :], other=0.0
        ).to(tl.float32)
        logits += tl.sum(q_pe_chunk[None, :] * kp_chunk, axis=1)

        # Scale and mask invalid
        logits = logits * sm_scale
        logits = tl.where(valid_mask, logits, -float("inf"))

        # ---- Online softmax update ----
        m_new = tl.maximum(m_i, tl.max(logits, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new)
        p = tl.where(valid_mask, p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha

        # ---- Accumulate output tile [BLOCK_D_CKV] ----
        d_offs_out = d_out_start + tl.arange(0, BLOCK_D_CKV)
        kc_addrs = indices_i64[:, None] * HEAD_DIM_CKV + d_offs_out[None, :]
        kc_chunk = tl.load(
            ckv_flat_ptr + kc_addrs,
            mask=valid_mask[:, None], other=0.0
        ).to(tl.float32)
        acc += tl.sum(p[:, None] * kc_chunk, axis=0)

        m_i = m_new
        l_i = l_new

    # Finalize: acc / l_i
    acc = acc / l_i

    # Write output tile
    d_offs_out = d_out_start + tl.arange(0, BLOCK_D_CKV)
    out_ptrs = output_ptr + token_id * stride_o_t + head_id * stride_o_h + d_offs_out
    tl.store(out_ptrs, acc.to(tl.bfloat16))

    # Write LSE (only from d_tile_id == 0 to avoid race condition)
    if d_tile_id == 0:
        # LSE in base 2: log2(sum(exp(logits_scaled)))
        # = log2(l_i * exp(m_i))
        # = log2(l_i) + m_i * log2(e)
        LOG2E: tl.constexpr = 1.4426950408889634
        lse_val = tl.log2(l_i) + m_i * LOG2E
        lse_ptr_out = lse_ptr + token_id * stride_lse_t + head_id * stride_lse_h
        tl.store(lse_ptr_out, lse_val)


# ============================================================================
# Host function (DPS style)
# ============================================================================
def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    """
    Sparse attention kernel (DPS style).

    Args:
        q_nope: [num_tokens, 16, 512] bfloat16
        q_pe: [num_tokens, 16, 64] bfloat16
        ckv_cache: [num_pages, 64, 512] bfloat16
        kpe_cache: [num_pages, 64, 64] bfloat16
        sparse_indices: [num_tokens, 2048] int32
        sm_scale: float32 scalar
        output: [num_tokens, 16, 512] bfloat16 (pre-allocated)
        lse: [num_tokens, 16] float32 (pre-allocated)
    """
    num_tokens = q_nope.shape[0]
    HEAD_DIM_CKV = 512
    HEAD_DIM_KPE = 64
    TOPK = 2048
    NUM_QO_HEADS = 16

    # Flatten caches to [total_kv_tokens, dim] for gather
    ckv_flat = ckv_cache.reshape(-1, HEAD_DIM_CKV)
    kpe_flat = kpe_cache.reshape(-1, HEAD_DIM_KPE)

    # Tile configuration
    BLOCK_KV = 128
    BLOCK_D_CKV = 64
    BLOCK_D_KPE = 64  # >= HEAD_DIM_KPE so covers full KPE in one tile
    NUM_D_TILES = HEAD_DIM_CKV // BLOCK_D_CKV  # 8

    grid = (num_tokens, NUM_QO_HEADS, NUM_D_TILES)

    _sparse_attn_dtile_kernel[grid](
        q_nope, q_pe,
        ckv_flat, kpe_flat,
        sparse_indices,
        output, lse,
        sm_scale,
        num_tokens,
        TOPK,
        # Q strides
        q_nope.stride(0), q_nope.stride(1),
        q_pe.stride(0), q_pe.stride(1),
        # Output strides
        output.stride(0), output.stride(1),
        lse.stride(0), lse.stride(1),
        # Constexprs
        BLOCK_KV=BLOCK_KV,
        BLOCK_D_CKV=BLOCK_D_CKV,
        BLOCK_D_KPE=BLOCK_D_KPE,
        NUM_D_TILES=NUM_D_TILES,
        HEAD_DIM_CKV=HEAD_DIM_CKV,
        HEAD_DIM_KPE=HEAD_DIM_KPE,
        TOPK=TOPK,
    )
