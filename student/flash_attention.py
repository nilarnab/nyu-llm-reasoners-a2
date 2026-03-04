import math

from math import ceil

import torch
from einops import rearrange

class FlashAttentionNT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, Bq=16, Bk=16):
        print("Context", ctx)
        print("Shape", Q.shape, K.shape, V.shape)

        Nq = Q.shape[1]
        d = Q.shape[2]

        Nk = K.shape[1]

        Tq = ceil(Nq/Bq)
        Q_split = rearrange(Q, 'a (r b) c ->  r a b c', r=Tq)

        Tk = ceil(Nk/Bk)
        K_split = rearrange(K, 'a (r b) c ->  r a b c', r=Tk)
        V_split = rearrange(V, 'a (r b) c ->  r a b c', r=Tk)

        O_tiles = []
        L_tiles = []
        for i in range(Tq):
            Q_i = Q_split[i]
            # O_i = torch.zeros(Bq, d)

            batch_size = Q.shape[0]

            O_i = torch.zeros(batch_size, Bq, d, device=Q.device)
            l_i_0 = torch.zeros(batch_size, Bq, device=Q.device)
            # m_i_0 = torch.full((batch_size, Bq), float('-inf'), device=Q.device)

            # l_i_0 = torch.zeros(Bq)
            # m_i_0 = torch.full((Bq,), float('-inf'))

            mi_j = torch.full((batch_size, Bq), -1e9, device=Q.device)  # <- not -inf
            l_i_j = torch.zeros(batch_size, Bq, device=Q.device)
            O_i = torch.zeros(batch_size, Bq, d, device=Q.device)

            # mi_j = m_i_0
            # l_i_j = l_i_0

            print("l_i_j mi_j original", l_i_j, "mi_j", mi_j)

            for j in range(Tk):
                k_j = K_split[j]
                v_j = V_split[j]

                print("Q_i shape:", Q_i.shape)
                print("K_j shape:", k_j.shape)

                # S_i_j = torch.matmul(Q_i, k_j.transpose(-1, -2)) / d**0.5
                S_i_j = torch.matmul(Q_i, k_j.transpose(-1, -2)) / d ** 0.5
                S_i_j = torch.clamp(S_i_j, -1e9, 1e9)
                print("S i j shape", S_i_j.shape)

                print("mi_j", mi_j, "S_i_j.max(dim=2).values", S_i_j.max(dim=2).values)
                mi_j_new = torch.maximum(mi_j, S_i_j.max(dim=2).values)
                print("mi_j_new now", mi_j_new)

                P_tilde_i_j = torch.exp(S_i_j - mi_j_new.unsqueeze(-1))

                print("mi_j_new", mi_j_new, "mi_j", mi_j, "l_i_j", l_i_j, "P_tilde_i_j.sum(dim=2)", P_tilde_i_j.sum(dim=2))
                # l_i_j_new = torch.exp(mi_j_new - mi_j) * l_i_j + P_tilde_i_j.sum(dim=2)
                l_i_j_new = torch.exp(mi_j - mi_j_new) * l_i_j + P_tilde_i_j.sum(dim=2)
                print("l_i_j_new after operation", l_i_j_new)

                O_i = torch.exp(mi_j - mi_j_new).unsqueeze(-1) * O_i + P_tilde_i_j @ v_j

                print("Oi", O_i)

                mi_j = mi_j_new
                l_i_j = l_i_j_new

                print("l_i_j", l_i_j)

            # print("l_i_j.unsqueeze(-1)", l_i_j.unsqueeze(-1))
            # O_i = O_i / l_i_j.unsqueeze(-1)
            #
            # print("Oi changed", O_i)
            # l_i = mi_j + torch.log(l_i_j)

            # print("li", l_i)

            eps = 1e-12
            O_i = O_i / (l_i_j.unsqueeze(-1) + eps)
            l_i = mi_j + torch.log(l_i_j + eps)

            O_tiles.append(O_i)
            L_tiles.append(l_i)

        O = torch.cat(O_tiles, dim=1)
        L = torch.cat(L_tiles, dim=1)
        print("L shape", L.shape)

        ctx.save_for_backward(L)

        return O


    @staticmethod
    def backward():
        return "Nothing"
