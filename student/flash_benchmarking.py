import torch
import triton
from a1_basics import model as basic_model

from student.triton_flash_attention import FlashAttention
from student.utils import create_latex_table

seq_length = 128
embed_dim = 16
PRECISIONS = [torch.float32, torch.bfloat16]
batch_size = 1
is_causal  = True


def pytorch_end_to_end(q, k, v, do, is_causal):
    output_normal = basic_model.scaled_dot_product_attention(q, k, v, is_causal)
    output_normal.backward(do, retain_graph=True)


def flashattention_end_to_end(q, k, v, do, is_causal):
    output_flash_triton = FlashAttention.apply(q, k, v, is_causal, q_tile_size, k_tile_size)
    output_flash_triton.backward(do, retain_graph=True)

    pass


rows1 = []
rows2 = []
while seq_length <= 65536:
    while embed_dim <= 128:
        for precision in PRECISIONS:
            q_tile_size = 16
            k_tile_size = 16

            q = torch.randn(batch_size, seq_length, embed_dim, device="cuda", dtype=precision, requires_grad=True)
            k = torch.randn(batch_size, seq_length, embed_dim, device="cuda", dtype=precision, requires_grad=True)
            v = torch.randn(batch_size, seq_length, embed_dim, device="cuda", dtype=precision, requires_grad=True)
            do = torch.randn(batch_size, seq_length, embed_dim, device="cuda", dtype=precision)

            normal_fwd_time = triton.testing.do_bench(
                lambda: basic_model.scaled_dot_product_attention(q, k, v, is_causal)
            )

            output_normal = basic_model.scaled_dot_product_attention(q, k, v, is_causal)
            normal_bwd_ms = triton.testing.do_bench(
                lambda: output_normal.backward(do, retain_graph=True)
            )

            normal_full_ms = triton.testing.do_bench(
                lambda: pytorch_end_to_end(q, k, v, do, is_causal)
            )


            # now doing the triton part
            flash_attention_triton_fwd_time = triton.testing.do_bench(
                lambda: FlashAttention.apply(q, k, v, is_causal, q_tile_size, k_tile_size)
            )

            output_flash_triton = FlashAttention.apply(q, k, v, is_causal, q_tile_size, k_tile_size)
            flash_attention_triton_bwd_time = triton.testing.do_bench(
                lambda: output_flash_triton.backward(do, retain_graph=True)
            )

            flash_attention_triton_full_time = triton.testing.do_bench(
                lambda: flashattention_end_to_end(q, k, v, do, is_causal)
            )

            rows1.append([
                seq_length, embed_dim, str(precision),
                round(normal_fwd_time,2),
                round(normal_bwd_ms, 2),
                round(normal_full_ms, 2),
            ])

            rows2.append([
                seq_length, embed_dim, str(precision),
                round(normal_fwd_time, 2),
                round(normal_bwd_ms, 2),
                round(normal_full_ms, 2),
            ])

        embed_dim = embed_dim * 2

    seq_length = seq_length * 2


latex_table_string_normal = create_latex_table(
        [
            'SeqLength', 'D', 'precision', 'Forward (ms)', 'Backward (ms)', 'Full (ms)',
        ],
    rows1
    )

latex_table_string_flash_attention = create_latex_table(
        [
            'SeqLength', 'D', 'precision', 'Forward (ms)', 'Backward (ms)', 'Full (ms)',
        ],
    rows2
    )

print("Latex Table Normal")
print(latex_table_string_normal)

print("Latex Table Flash Attention")
print(latex_table_string_flash_attention)














