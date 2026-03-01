import argparse
import os

from student.defaults import MACHINE, SWEEPS, DEVICE
from student.measurable_training_loop import training_loop, eval_timing_loop
import numpy as np
from student.utils import create_latex_table
import wandb

os.environ["WANDB_API_KEY"] = "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"
wandb.login(key=os.environ["WANDB_API_KEY"])

CONTEXT_LENGTHS = [256]
SIZES = ["small", "medium", "large", "xl", "2.7B"]

def trigger_loop(
        use_mixed_precision=True,
        eval_mode = False,
        use_compiled = False,
        profile_memory = False,
):
    wandb.init(
        project=f"assignment-2-{MACHINE}",
        name=f"memory_profiling",
        config={
            "model": "transformer",
        }
    )

    table_val = []

    for size_key in SIZES:
        for context_length in CONTEXT_LENGTHS:
            print("TRIGGERING FOR CONTEXT LENGTH", context_length)
            filename = f"{size_key}_{str(context_length)}_eval{str(eval_mode)}_mixed_prec{str(use_mixed_precision)}.pickle"
            location = f"memory_profiles/{filename}"
            if eval_mode:
                res = eval_timing_loop(
                    d_model=SWEEPS[size_key]['d_model'],
                    d_ff=SWEEPS[size_key]['d_ff'],
                    num_layers=SWEEPS[size_key]['num_layers'],
                    num_heads=SWEEPS[size_key]['num_heads'],
                    use_mixed_precision=use_mixed_precision,
                    context_length=context_length,
                    profile_memory=profile_memory,
                    profile_memory_location=location,
                    use_compiled=use_compiled,
                )

            else:
                res = training_loop(
                    d_model=SWEEPS[size_key]['d_model'],
                    d_ff=SWEEPS[size_key]['d_ff'],
                    num_layers=SWEEPS[size_key]['num_layers'],
                    num_heads=SWEEPS[size_key]['num_heads'],
                    use_mixed_precision=use_mixed_precision,
                    context_length=context_length,
                    profile_memory=profile_memory,
                    profile_memory_location=location,
                    use_compiled=use_compiled
                )

            row = [size_key, context_length]
            if 'FORWARD_PASS_TIME' in res:
                row.append(round(np.mean(res['FORWARD_PASS_TIME']) * 1000, 2))
            if 'BACKWARD_PASS_TIME' in res:
                row.append(round(np.mean(res['BACKWARD_PASS_TIME']) * 1000, 2))
            if 'OPTIMIZER_TIME' in res:
                row.append(round(np.mean(res['OPTIMIZER_TIME']) * 1000, 2))
            if 'FULL_TRAIN_TIME' in res:
                row.append(round(np.mean(res['FULL_TRAIN_TIME']) * 1000, 2))


            table_val.append([el for el in row])

            if not eval_mode:
                headers = ['Size', 'Context Length', 'Forward Pass Mean (ms)']
            else:
                headers = ['Size', 'Context Length', 'Forward Mean (ms)', 'Backward Mean (ms)', 'Optimizer Mean (ms)',
                           'Full Pass (ms)']

            latex_table_string = create_latex_table(headers, table_val)

            print("partial")
            print(latex_table_string)


    if eval_mode:
        headers = ['Size', 'Context Length', 'Forward Pass Mean (ms)']
    else:
        headers = ['Size', 'Context Length', 'Forward Mean (ms)', 'Backward Mean (ms)', 'Optimizer Mean (ms)', 'Full Pass (ms)']

    latex_table_string = create_latex_table(headers, table_val)

    print("complete")
    print(latex_table_string)

    wandb.finish()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_mixed_precision', type=str, default="FALSE")
    parser.add_argument('--use_compiled', type=str, default="FALSE")
    parser.add_argument('--eval_mode', type=str, default="FALSE")
    parser.add_argument('--profile_memory', type=str, default="FALSE")

    args = parser.parse_args()

    trigger_loop(
        use_mixed_precision= (args.use_mixed_precision == 'TRUE'),
        eval_mode = (args.eval_mode == 'TRUE'),
        use_compiled= (args.use_compiled == 'TRUE'),
        profile_memory= (args.profile_memory == 'TRUE')
    )
