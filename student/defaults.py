import os

import torch

VOCAB_SIZE = 10000
BATCH_SIZE = 4
CONTEXT_LENGTH = 256
ITERATIONS = 5000
WARMUP_ITERS = min(1000, int(ITERATIONS * 0.25))
COSINE_CYCLE_ITERS = ITERATIONS
MIN_LR_RATIO = 0.1
D_MODEL = 512
NUM_BLOCKS = 4
NUM_HEADS = 16
D_FF = int(1344 * (8/3))
ROPE_THETA = 10000.0

BETAS=(0.9, 0.999)
WEIGHT_DECAY = 0.01

MAX_LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = MAX_LEARNING_RATE * MIN_LR_RATIO

if torch.cuda.is_available():
    print("device set to CUDA")
    DEVICE = "cuda"
elif torch.mps.is_available():
    print("device set to MPS")
    DEVICE = "mps"
else:
    print("no gpu available, defaulting to CPU")
    DEVICE = "cpu"



# TIME MEASRUEMENT
WARMUP_COUNT_BEFORE_TIME = 5
MEASURE_FOR_COUNT = 10



MACHINE = os.getenv("MACHINE", "hpc")
print("MACHINE", MACHINE)

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