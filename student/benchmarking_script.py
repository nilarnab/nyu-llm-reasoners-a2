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

def ques_b(use_mixed_precision=True):
    wandb.init(
        project=f"assignment-2-{MACHINE}",
        name=f"benchmarking_script_ques_b",
        config={
            "model": "transformer",
        }
    )

    time_measure_params = {
        "warmup_count": 5,
        "measure_for_count": 10
    }

    table_val = []



    for size_key in SWEEPS:
        print("using size now", size_key)
        res = training_loop(
            d_model=SWEEPS[size_key]['d_model'],
            d_ff=SWEEPS[size_key]['d_ff'],
            num_layers=SWEEPS[size_key]['num_layers'],
            num_heads=SWEEPS[size_key]['num_heads'],
            time_measure_params=time_measure_params,
            use_mixed_precision=use_mixed_precision
        )

        print("SIZE", size_key, "RES", res)
        mean_val_fwd = np.mean(res['FORWARD_PASS_TIME'])
        std_val_fwd = np.std(res['FORWARD_PASS_TIME'], ddof=1)

        mean_val_bwd = np.mean(res['BACKWARD_PASS_TIME'])
        std_val_bwd = np.std(res['BACKWARD_PASS_TIME'], ddof=1)

        table_val.append(
            [size_key, mean_val_fwd, std_val_fwd, mean_val_bwd, std_val_bwd]
        )

        table_latex_string = create_latex_table(
             ['Size', 'Forward Pass Mean', 'Forward Pass Standard deviation', 'Backward Pass Mean',
              'Backward Pass Standard deviation'],
             table_val
         )

        print("partial", table_val)
        print(table_latex_string)

    print("COMPLETE RESULT:")
    table_latex_string = create_latex_table(
        ['Size', 'Forward Pass Mean', 'Forward Pass Standard deviation', 'Backward Pass Mean',
         'Backward Pass Standard deviation'],
        table_val
    )
    print(table_latex_string)

    wandb.finish()

def ques_c():
    wandb.init(
        project=f"assignment-2-{MACHINE}",
        name=f"benchmarking_script_ques_c",
        config={
            "model": "transformer",
        }
    )

    warmup_vals = [2]
    for warmup_val in warmup_vals:
        time_measure_params = {
            "warmup_count": warmup_val,
            "measure_for_count": 10
        }
        table_val = []
        for size_key in SWEEPS:
            res = training_loop(
                d_model=SWEEPS[size_key]['d_model'],
                d_ff=SWEEPS[size_key]['d_ff'],
                num_layers=SWEEPS[size_key]['num_layers'],
                num_heads=SWEEPS[size_key]['num_heads'],
                time_measure_params=time_measure_params
            )

            print("SIZE", size_key, "RES", res)
            mean_val_fwd = np.mean(res['FORWARD_PASS_TIME'])
            std_val_fwd = np.std(res['FORWARD_PASS_TIME'], ddof=1)

            mean_val_bwd = np.mean(res['BACKWARD_PASS_TIME'])
            std_val_bwd = np.std(res['BACKWARD_PASS_TIME'], ddof=1)

            table_val.append(
                [size_key, mean_val_fwd, std_val_fwd, mean_val_bwd, std_val_bwd]
            )
            
            print("WARMUP VAL", warmup_val)
            print("partial", table_val)
            table_latex_string = create_latex_table(
            ['Size', 'Forward Pass Mean', 'Forward Pass Standard deviation', 'Backward Pass Mean',
             'Backward Pass Standard deviation'],
            table_val
        )
            print(table_latex_string)

        print("COMPLETE RESULT: WARM UP", warmup_val)
        table_latex_string = create_latex_table(
            ['Size', 'Forward Pass Mean', 'Forward Pass Standard deviation', 'Backward Pass Mean',
             'Backward Pass Standard deviation'],
            table_val
        )
        print(table_latex_string)

        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_mixed_precision', type=str, default="TRUE")

    args = parser.parse_args()

    ques_b(use_mixed_precision=(args.use_mixed_precision == "TRUE"))
