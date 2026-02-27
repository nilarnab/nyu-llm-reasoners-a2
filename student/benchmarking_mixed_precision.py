import torch
from torch import nn
from utils import create_latex_table
from a1_basics import nn_utils as basic_nn_utils
from defaults import DEVICE


RESULTS = []

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        global RESULTS
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        global RESULTS
        outp_fc1 = self.fc1(x)
        RESULTS.append(["fc1(x) output", outp_fc1.dtype])
        x = self.relu(outp_fc1)
        x = self.ln(x)
        RESULTS.append(["Layer Norm output", x.dtype])
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    input = torch.rand(10, 10).to(DEVICE)
    target = torch.randint(0, 10, (10, ), dtype=torch.long).to(DEVICE)
    model = ToyModel(input.shape[1], 10).to(DEVICE)

    with torch.autocast(device_type=DEVICE, dtype=torch.float16):
        outp = model(input)
        RESULTS.append(["Model Output Logits", outp.dtype])

        loss = basic_nn_utils.cross_entropy(outp, target)
        RESULTS.append(["Loss type", loss.dtype])
        loss.backward()

        for name, param in model.named_parameters():
            RESULTS.append([f"Model parameter {name}", param.dtype])

        for name, param in model.named_parameters():
            RESULTS.append([f"Gradient of {name}", param.grad.dtype])

    latex_string = create_latex_table(['Variable', 'dtype'], RESULTS)
    print(latex_string)



