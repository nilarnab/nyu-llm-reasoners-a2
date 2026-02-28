import argparse
import os

from student.defaults import MACHINE, SWEEPS, DEVICE
from student.measurable_training_loop import training_loop, eval_timing_loop
import numpy as np
from student.utils import create_latex_table
import wandb

os.environ["WANDB_API_KEY"] = "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"
wandb.login(key=os.environ["WANDB_API_KEY"])

CONTEXT_LENGTHS = [128, 256, 512]
SIZE = "2.7B"

def trigger_loop(
        use_mixed_precision=True,
        eval_mode = False,
):
    wandb.init(
        project=f"assignment-2-{MACHINE}",
        name=f"memory_profiling",
        config={
            "model": "transformer",
        }
    )

    for context_length in CONTEXT_LENGTHS:
        print("TRIGGERING FOR CONTEXT LENGTH", context_length)
        filename = f"{SIZE}_{str(context_length)}_eval{str(eval_mode)}_mixed_prec{str(use_mixed_precision)}.pickle"
        location = f"memory_profiles/{filename}"
        size_key = SIZE
        if eval_mode:
            res = eval_timing_loop(
                d_model=SWEEPS[size_key]['d_model'],
                d_ff=SWEEPS[size_key]['d_ff'],
                num_layers=SWEEPS[size_key]['num_layers'],
                num_heads=SWEEPS[size_key]['num_heads'],
                use_mixed_precision=use_mixed_precision,
                context_length=context_length,
                profile_memory=True,
                profile_memory_location=location
            )

        else:
            res = training_loop(
                d_model=SWEEPS[size_key]['d_model'],
                d_ff=SWEEPS[size_key]['d_ff'],
                num_layers=SWEEPS[size_key]['num_layers'],
                num_heads=SWEEPS[size_key]['num_heads'],
                use_mixed_precision=use_mixed_precision,
                profile_memory_location=location,
                context_length=context_length
            )

    wandb.finish()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_mixed_precision', type=str, default="TRUE")
    parser.add_argument('--eval_mode', type=str, default="FALSE")

    args = parser.parse_args()

    trigger_loop(
        use_mixed_precision= (args.use_mixed_precision == 'TRUE'),
        eval_mode = (args.eval_mode == 'TRUE'),
    )
