"""
Sparse Attention Kernel v2 (DSA) for FlashInfer MLSys 2026 Contest.

Track: sparse_attention (DeepSeek-V3.2 Native Sparse Attention)
Definition: dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
Author: DiegoCao

V2 optimizations over V1:
  - exp2-based online softmax (fast math, avoids ln2 multiply)
  - BLOCK_D=128 → 4 d-tiles instead of 8 (halves logits redundancy)
  - Pre-loaded q_pe vector (reused every KV chunk iteration)
  - BLOCK_KV=256 for fewer loop iterations (8 iters vs 16)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_attn_kernel(
    q_nope_ptr, q_pe_ptr,
    ckv_flat_ptr, kpe_flat_ptr,
    sparse_indices_ptr,
    output_ptr, lse_ptr,
    sm_scale_log2,  # sm_scale * log2(e)
    num_tokens,
    stride_qn_t, stride_qn_h,
    stride_qp_t, stride_qp_h,
    stride_o_t, stride_o_h,
    stride_lse_t, stride_lse_h,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_D_TILES: tl.constexpr,
):
    """
    One program per (token, head, d_tile).
    Logits computed once per KV chunk (tiled over CKV+KPE dims).
    Output accumulated for this d_tile only.
    Uses exp2 for fast softmax.
    """
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    d_tile_id = tl.program_id(2)

    d_out_start = d_tile_id * BLOCK_D
    si_base = sparse_indices_ptr + token_id * TOPK

    # Online softmax state
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Pre-load q_pe (64-dim, stays in registers across all KV chunks)
    d_offs_kpe = tl.arange(0, HEAD_DIM_KPE)
    q_pe_vec = tl.load(
        q_pe_ptr + token_id * stride_qp_t + head_id * stride_qp_h + d_offs_kpe
    ).to(tl.float32)

    for kv_start in range(0, TOPK, BLOCK_KV):
        kv_offs = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offs < TOPK

        indices = tl.load(si_base + kv_offs, mask=kv_mask, other=-1)
        valid_mask = (indices != -1) & kv_mask
        indices_i64 = indices.to(tl.int64)

        # ---- Compute logits [BLOCK_KV] ----
        logits = tl.zeros([BLOCK_KV], dtype=tl.float32)

        # CKV dot product (tiled over head_dim in BLOCK_D chunks)
        for d_start in tl.static_range(0, HEAD_DIM_CKV, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            q_chunk = tl.load(
                q_nope_ptr + token_id * stride_qn_t + head_id * stride_qn_h + d_offs
            ).to(tl.float32)
            k_addrs = indices_i64[:, None] * HEAD_DIM_CKV + d_offs[None, :]
            k_chunk = tl.load(
                ckv_flat_ptr + k_addrs,
                mask=valid_mask[:, None], other=0.0
            ).to(tl.float32)
            logits += tl.sum(q_chunk[None, :] * k_chunk, axis=1)

        # KPE dot product (single tile, 64-dim)
        kp_addrs = indices_i64[:, None] * HEAD_DIM_KPE + d_offs_kpe[None, :]
        kp_chunk = tl.load(
            kpe_flat_ptr + kp_addrs,
            mask=valid_mask[:, None], other=0.0
        ).to(tl.float32)
        logits += tl.sum(q_pe_vec[None, :] * kp_chunk, axis=1)

        # Scale (in log2 space) and mask
        logits = logits * sm_scale_log2
        logits = tl.where(valid_mask, logits, -float("inf"))

        # ---- Online softmax with exp2 ----
        m_new = tl.maximum(m_i, tl.max(logits, axis=0))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(logits - m_new)
        p = tl.where(valid_mask, p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha

        # ---- Accumulate output d_tile [BLOCK_D] ----
        d_offs_out = d_out_start + tl.arange(0, BLOCK_D)
        kc_addrs = indices_i64[:, None] * HEAD_DIM_CKV + d_offs_out[None, :]
        kc_chunk = tl.load(
            ckv_flat_ptr + kc_addrs,
            mask=valid_mask[:, None], other=0.0
        ).to(tl.float32)
        acc += tl.sum(p[:, None] * kc_chunk, axis=0)

        m_i = m_new
        l_i = l_new

    # Finalize
    acc = acc / l_i

    # Write output tile
    d_offs_out = d_out_start + tl.arange(0, BLOCK_D)
    out_ptrs = output_ptr + token_id * stride_o_t + head_id * stride_o_h + d_offs_out
    tl.store(out_ptrs, acc.to(tl.bfloat16))

    # Write LSE (only from d_tile_id == 0)
    if d_tile_id == 0:
        # LSE in base 2: log2(l_i) + m_i (m_i already in log2 space)
        lse_val = tl.log2(l_i) + m_i
        tl.store(lse_ptr + token_id * stride_lse_t + head_id * stride_lse_h, lse_val)


def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    num_tokens = q_nope.shape[0]
    HEAD_DIM_CKV = 512
    HEAD_DIM_KPE = 64
    TOPK = 2048
    NUM_QO_HEADS = 16
    LOG2E = 1.4426950408889634

    ckv_flat = ckv_cache.reshape(-1, HEAD_DIM_CKV)
    kpe_flat = kpe_cache.reshape(-1, HEAD_DIM_KPE)
    sm_scale_log2 = sm_scale * LOG2E

    BLOCK_KV = 256

    # Adaptive BLOCK_D: more d_tiles for small batches (parallelism),
    # fewer for large batches (less redundant logits computation)
    if num_tokens <= 2:
        BLOCK_D = 64
    else:
        BLOCK_D = 128
    NUM_D_TILES = HEAD_DIM_CKV // BLOCK_D

    grid = (num_tokens, NUM_QO_HEADS, NUM_D_TILES)

    _sparse_attn_kernel[grid](
        q_nope, q_pe,
        ckv_flat, kpe_flat,
        sparse_indices,
        output, lse,
        sm_scale_log2, num_tokens,
        q_nope.stride(0), q_nope.stride(1),
        q_pe.stride(0), q_pe.stride(1),
        output.stride(0), output.stride(1),
        lse.stride(0), lse.stride(1),
        BLOCK_KV=BLOCK_KV, BLOCK_D=BLOCK_D,
        HEAD_DIM_CKV=HEAD_DIM_CKV, HEAD_DIM_KPE=HEAD_DIM_KPE,
        TOPK=TOPK, NUM_D_TILES=NUM_D_TILES,
    )
