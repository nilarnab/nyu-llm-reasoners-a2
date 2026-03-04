from typing import Any

import math

from math import ceil

import torch
from einops import rearrange

import triton
import triton.language as tl
from triton import cdiv

from student.defaults import DEVICE


@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Nq = N_QUERIES
    Nk = N_KEYS

    # Tk = ceil(Nk / Q_TILE_SIZE)

    # TODO: not sure how boundary check works
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    mi_j = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l_i_j = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S_i_j = tl.dot(Q_i, tl.trans(k_j)) * scale
        mi_j_new = tl.maximum(mi_j, tl.max(S_i_j, axis=1))

        # TODO: What happened here?
        P_tilde_i_j = tl.exp(S_i_j - mi_j_new[:, None])
        l_i_j_new = tl.exp(mi_j - mi_j_new) * l_i_j + tl.sum(P_tilde_i_j, axis=1)
        O_i = tl.exp(mi_j - mi_j_new)[:, None] * O_i + tl.dot(P_tilde_i_j, v_j)

        mi_j = mi_j_new
        l_i_j = l_i_j_new

        # --------
        # advancing
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))

        # TODO: how is this advance happening in row wise?
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    eps = 1e-12
    O_i = O_i / (l_i_j[:, None] + eps)
    l_i = mi_j + tl.log(l_i_j + eps)

    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, l_i, boundary_check=(0,))


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, Bq=16, Bk=16) -> Any:
        b, Nq, D = q.shape
        Nk = k.shape[1]
        scale = 1.0 / (D ** 0.5)

        O = torch.empty(b, Nq, D, device=q.device, dtype=q.dtype)
        L = torch.empty(b, Nq, device=q.device, dtype=torch.float32)

        Tq = cdiv(Nq, Bq)
        grid = (Tq, b)

        # TODO: interestig here how grid is launched
        flash_fwd_kernel[grid](
            q, k, v,
            O, L,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            scale,
            D=D,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
        )

        ctx.save_for_backward(q, k, v, O, L)

        return O


    @staticmethod
    def backward():
        return "Nothing"
