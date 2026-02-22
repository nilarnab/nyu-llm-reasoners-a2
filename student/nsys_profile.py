import argparse
import os

from student.defaults import MACHINE
from student.measurable_training_loop import training_loop
import numpy as np
from student.utils import create_latex_table
import wandb


os.environ["WANDB_API_KEY"] = "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"
wandb.login(key=os.environ["WANDB_API_KEY"])


SWEEPS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25,
    },
    "2.7B": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
}


def run_tests(size_key, context_size):
    print(f"RUNNING SIZE {size_key}, CONTEXT_SIZE: {context_size}")
    wandb.init(
        project=f"assignment-2-{MACHINE}",
        name=f"nsys_profiling",
        config={
            "model": "transformer",
        }
    )

    time_measure_params = {
        "warmup_count": 5,
        "measure_for_count": 10
    }

    res = training_loop(
            d_model=SWEEPS[size_key]['d_model'],
            d_ff=SWEEPS[size_key]['d_ff'],
            num_layers=SWEEPS[size_key]['num_layers'],
            num_heads=SWEEPS[size_key]['num_heads'],
            time_measure_params=time_measure_params
        )

    print("result, not used", res)

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=int, required=True)
    parser.add_argument('--context_size', type=int, required=True)

    args = parser.parse_args()

    run_tests(args.model_size, args.context_size)
