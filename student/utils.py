import torch
import numpy.typing as npt
import numpy as np

def conditionally_torch_sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()
        return

    print("Tried to cuda.synchronize when not in cuda, but in", device)

def data_loader(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    original_tensor = torch.Tensor(dataset.copy())
    length = original_tensor.shape[0]
    start_indices = np.random.randint(0, length - context_length, batch_size)

    inputs = []
    outputs = []

    for start_index in start_indices:
        inputs.append(original_tensor[start_index: start_index + context_length])
        outputs.append(original_tensor[start_index + 1: start_index + context_length + 1])

    input_tensor = torch.stack(inputs).long().to(device)
    output_tensor = torch.stack(outputs).long().to(device)

    return input_tensor, output_tensor

def create_latex_table(headers, entries, caption=None, label=None):
    num_cols = len(headers)
    col_format = " | ".join(["c"] * num_cols)
    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{| {col_format} |}}")
    lines.append("\\hline")

    header_row = " & ".join(str(h) for h in headers) + " \\\\"
    lines.append(header_row)
    lines.append("\\hline")

    for row in entries:
        row_line = " & ".join(str(item) for item in row) + " \\\\"
        lines.append(row_line)

    lines.append("\\hline")
    lines.append("\\end{tabular}")

    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")

    lines.append("\\end{table}")

    return "\n".join(lines)