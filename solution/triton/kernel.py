"""
Gated Delta Net (GDN) Kernels for FlashInfer MLSys 2026 Contest.

Tracks:
  - gdn_decode_qk4_v8_d128_k_last (batch=1, T=1)
  - gdn_prefill_qk4_v8_d128_k_last (variable T up to 8192)

Constants: num_q_heads=4, num_k_heads=4, num_v_heads=8, head_size=128
GVA mapping: v_head h -> q/k_head h // 2
State layout: k-last [N, H_V, V, K]
"""

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# =====================================================================
# GDN Decode Triton Kernel
# =====================================================================

@triton.jit
def _gdn_decode_kernel(
    q_ptr, k_ptr, v_ptr,
    state_ptr, output_ptr, new_state_ptr,
    g_ptr, beta_ptr,
    scale,
    stride_qk_b, stride_qk_h,
    stride_v_b, stride_v_h,
    stride_s_b, stride_s_h, stride_s_v,
    stride_o_b, stride_o_h,
    BLOCK_V: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    b_idx = pid_bh // NUM_V_HEADS
    h_v = pid_bh % NUM_V_HEADS
    h_qk = h_v * NUM_Q_HEADS // NUM_V_HEADS

    v_start = pid_v * BLOCK_V
    v_offs = v_start + tl.arange(0, BLOCK_V)
    k_offs = tl.arange(0, HEAD_SIZE)

    g_val = tl.load(g_ptr + b_idx * NUM_V_HEADS + h_v)
    beta_val = tl.load(beta_ptr + b_idx * NUM_V_HEADS + h_v)

    k_vec = tl.load(k_ptr + b_idx * stride_qk_b + h_qk * stride_qk_h + k_offs).to(tl.float32)
    q_vec = tl.load(q_ptr + b_idx * stride_qk_b + h_qk * stride_qk_h + k_offs).to(tl.float32)
    v_tile = tl.load(v_ptr + b_idx * stride_v_b + h_v * stride_v_h + v_offs).to(tl.float32)

    s_base = state_ptr + b_idx * stride_s_b + h_v * stride_s_h
    s_offsets = v_offs[:, None] * stride_s_v + k_offs[None, :]
    state_tile = tl.load(s_base + s_offsets)

    g_state = g_val * state_tile
    old_v = tl.sum(g_state * k_vec[None, :], axis=1)
    delta_v = beta_val * (v_tile - old_v)
    new_state_tile = g_state + delta_v[:, None] * k_vec[None, :]

    ns_base = new_state_ptr + b_idx * stride_s_b + h_v * stride_s_h
    tl.store(ns_base + s_offsets, new_state_tile)

    output_v = scale * tl.sum(new_state_tile * q_vec[None, :], axis=1)
    o_base = output_ptr + b_idx * stride_o_b + h_v * stride_o_h
    tl.store(o_base + v_offs, output_v.to(tl.bfloat16))


def gdn_decode(q, k, v, state, A_log, a, dt_bias, b, scale):
    B, T, num_q_heads, K = q.shape
    num_v_heads = v.shape[2]
    V = v.shape[3]
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x)).squeeze(1)
    beta = torch.sigmoid(b.float()).squeeze(1)

    q_2d = q.squeeze(1).contiguous()
    k_2d = k.squeeze(1).contiguous()
    v_2d = v.squeeze(1).contiguous()

    if state is None:
        state = torch.zeros(B, num_v_heads, V, K, dtype=torch.float32, device=device)

    output = torch.empty(B, num_v_heads, V, dtype=torch.bfloat16, device=device)
    new_state = torch.empty_like(state)

    BLOCK_V = 32
    NUM_V_TILES = V // BLOCK_V
    grid = (B * num_v_heads, NUM_V_TILES)

    _gdn_decode_kernel[grid](
        q_2d, k_2d, v_2d,
        state, output, new_state,
        g, beta, scale,
        q_2d.stride(0), q_2d.stride(1),
        v_2d.stride(0), v_2d.stride(1),
        state.stride(0), state.stride(1), state.stride(2),
        output.stride(0), output.stride(1),
        BLOCK_V=BLOCK_V, HEAD_SIZE=K,
        NUM_V_HEADS=num_v_heads, NUM_Q_HEADS=num_q_heads,
    )

    output = output.unsqueeze(1)
    return output, new_state


def _matmul(a, b):
    return a.float() @ b.float()


def gdn_prefill(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """GDN prefill — exact copy of reference implementation."""
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    # Compute g and beta from raw parameters
    x = a.float() + dt_bias.float()  # [total_seq_len, HV]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [total_seq_len, HV]
    beta = torch.sigmoid(b.float())  # [total_seq_len, HV]

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    output = torch.zeros(
        (total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=device
    )
    new_state = torch.zeros(
        (num_seqs, num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
    )

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start

        if seq_len <= 0:
            continue

        if state is not None:
            state_HKV = state[seq_idx].clone().float().transpose(-1, -2)  # [H,V,K] -> [H,K,V]
        else:
            state_HKV = torch.zeros(
                (num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
            )

        for i in range(seq_len):
            t = seq_start + i
            q_H1K = q_exp[t].unsqueeze(1).float()
            k_H1K = k_exp[t].unsqueeze(1).float()
            v_H1V = v[t].unsqueeze(1).float()
            g_H11 = g[t].unsqueeze(1).unsqueeze(2)
            beta_H11 = beta[t].unsqueeze(1).unsqueeze(2)

            old_state_HKV = g_H11 * state_HKV
            old_v_H1V = _matmul(k_H1K, old_state_HKV)
            new_v_H1V = beta_H11 * v_H1V + (1 - beta_H11) * old_v_H1V
            state_remove = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), old_v_H1V)
            state_update = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), new_v_H1V)
            state_HKV = old_state_HKV - state_remove + state_update

            o_H1V = scale * _matmul(q_H1K, state_HKV)
            output[t] = o_H1V.squeeze(1).to(torch.bfloat16)

        new_state[seq_idx] = state_HKV.transpose(-1, -2)  # [H,K,V] -> [H,V,K]

    return output, new_state


# =====================================================================
# Entry Point
# =====================================================================

def kernel(q, k, v, state, A_log, a, dt_bias, b, *args):
    if len(args) == 1:
        scale = args[0]
        return gdn_decode(q, k, v, state, A_log, a, dt_bias, b, scale)
    elif len(args) == 2:
        cu_seqlens, scale = args
        return gdn_prefill(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
    else:
        raise ValueError(f"Expected 9 or 10 arguments, got {8 + len(args)}")
