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