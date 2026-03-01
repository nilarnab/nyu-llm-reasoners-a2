import argparse
import os
import timeit
from collections import defaultdict
from email.policy import default

import torch

from student.defaults import MACHINE, BATCH_SIZE, DEVICE
from student.measurable_training_loop import training_loop
from a1_basics import model as basic_model
import numpy as np

from student.utils import create_latex_table, conditionally_torch_sync
import wandb
os.environ["WANDB_API_KEY"] = "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"
wandb.login(key=os.environ["WANDB_API_KEY"])



D_MODELS = [16, 32, 64, 128]
CONTEXT_LENGTHS = [256, 1024, 4096, 8192, 16384]

WARMUP_COUNT = 10
MEASURE_COUNT = 100

def run_tests():
    wandb.init(
        project=f"assignment-2-{MACHINE}",
        name=f"benchmarking_script_ques_b",
        config={
            "model": "transformer",
        }
    )

    res = []
    for d_model in D_MODELS:
        for context_length in CONTEXT_LENGTHS:

            try:
                Q = torch.rand(8, context_length, d_model, device=DEVICE, requires_grad=True)
                K = torch.rand(8, context_length, d_model, device=DEVICE, requires_grad=True)
                V = torch.rand(8, context_length, d_model, device=DEVICE, requires_grad=True)

                # warmup
                print("Warming up")
                for it_id in range(WARMUP_COUNT):
                    out = basic_model.scaled_dot_product_attention(Q, K, V)
                    loss = out.sum()
                    loss.backward()
                    Q.grad = K.grad = V.grad = None

                reporter = defaultdict(lambda: [])
                reporter["FORWARD_TIME"] = []
                reporter["BACKWARD_TIME"] = []
                reporter["MEMORY_BEFORE_BACKWARD"] = []

                print("Running Measures")
                for it_id in range(MEASURE_COUNT):

                    forward_start_time = timeit.default_timer()
                    out = basic_model.scaled_dot_product_attention(Q, K, V)
                    conditionally_torch_sync(DEVICE)
                    reporter['FORWARD_PASS_TIME'].append(timeit.default_timer() - forward_start_time)

                    memory_before_backward = torch.cuda.memory_allocated()
                    reporter["MEMORY_BEFORE_BACKWARD"].append(memory_before_backward)

                    loss = out.sum()

                    backward_start_time = timeit.default_timer()
                    loss.backward()
                    conditionally_torch_sync(DEVICE)
                    reporter['BACKWARD_TIME'].append(timeit.default_timer() - backward_start_time)

                    Q.grad = K.grad = V.grad = None

                res.append([
                    d_model,
                    context_length,
                    round(np.mean(reporter['FORWARD_PASS_TIME']) * 1000, 2),
                    round(np.mean(reporter['BACKWARD_TIME']) * 1000, 2),
                    round(np.mean(reporter['MEMORY_BEFORE_BACKWARD']) / (1024 ** 2), 2)
                ])

                print("PARTIAL")
                latex_string = create_latex_table(
                    headers=["d_model", "context_length", "Forward Pass Mean Time", "Backward Pass Mean Time",
                             "Memory Before Backward"],
                    entries=res
                )
                print(latex_string)

            except RuntimeError as e:
                print("Errored out with", str(e))

    print("COMPLETE")
    latex_string = create_latex_table(
        headers=["d_model", "context_length", "Forward Pass Mean Time", "Backward Pass Mean Time", "Memory Before Backward"],
        entries=res
    )
    print(latex_string)

    wandb.finish()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--use_mixed_precision', type=str, default="TRUE")
    #
    # args = parser.parse_args()

    run_tests()

    # uv run student/benchmarking_script.py