import argparse
from collections import defaultdict
from contextlib import nullcontext
from email.policy import default
from pickletools import optimize
import timeit

from a1_basics import model as basic_model
from a1_basics import optimizer as basic_optimizer
from a1_basics import nn_utils as basic_nn_utils
from student.utils import data_loader, conditionally_torch_sync
from student.defaults import *
import torch
import numpy as np
import torch.cuda.nvtx as nvtx


def get_random_data(batch_size, context_length, vocab_size=VOCAB_SIZE, device=DEVICE):
    input_data = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device)
    target_data = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device)

    return input_data, target_data


def eval_timing_loop(
vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    batch_size=BATCH_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_BLOCKS,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    rope_theta=ROPE_THETA,
    device=DEVICE,
    time_measure_params=None,
    generate_data_randomly=True,
    train_data=None
):
    print("EVAL TIMING LOOP WITH:")
    print("vocab size", vocab_size, "context length", context_length)

    if time_measure_params is None:
        time_measure_params = {
            "warmup_count": WARMUP_COUNT_BEFORE_TIME,
            "measure_for_count": MEASURE_FOR_COUNT
        }

    if not generate_data_randomly:
        if train_data is None:
            raise Exception("Generate randomly is false, so train data cannot be None")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    model = basic_model.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )

    model.to(device)
    model.eval()
    print("model initialized in eval mode")

    res = defaultdict(lambda: [])

    # WARMUP PHASE
    with nvtx.range("WARM_UP"):
        with torch.no_grad():
            for it_id in range(time_measure_params['warmup_count']):
                print("warmup it_id", it_id)
                if generate_data_randomly:
                    input_tensor, target_tensor = get_random_data(
                        batch_size=batch_size, context_length=context_length, device=device
                    )
                else:
                    input_tensor, target_tensor = data_loader(train_data, batch_size, context_length, device)

                logits = model(input_tensor)
                conditionally_torch_sync(device)

    with torch.no_grad():
        for it_id in range(time_measure_params['measure_for_count']):
            with nvtx.range("FULL_EVAL_RUN"):
                # print("measuring it_id", it_id)

                if generate_data_randomly:
                    input_tensor, target_tensor = get_random_data(
                        batch_size=batch_size, context_length=context_length, device=device
                    )
                else:
                    input_tensor, target_tensor = data_loader(train_data, batch_size, context_length, device)

                forward_start_time = timeit.default_timer()
                with nvtx.range("FORWARD_PASS"):
                    logits = model(input_tensor)

                conditionally_torch_sync(device)
                res['FORWARD_PASS_TIME'].append(timeit.default_timer() - forward_start_time)

                loss = basic_nn_utils.cross_entropy(logits, target_tensor)


    return res


def training_loop(max_learning_rate=MAX_LEARNING_RATE,
                       min_learning_rate= MIN_LEARNING_RATE,
                        vocab_size = VOCAB_SIZE,
                        context_length = CONTEXT_LENGTH,
                        batch_size = BATCH_SIZE,
                        d_model = D_MODEL,
                        num_layers = NUM_BLOCKS,
                        num_heads = NUM_HEADS,
                        d_ff = D_FF,
                        rope_theta = ROPE_THETA,
                        device = DEVICE,
                  time_measure_params=None,
                  generate_data_randomly=True,
                  train_data=None,
                  use_homegrown_adam=False,
                    use_mixed_precision = True,
                       ):
    print("TRINING LOOP WITH:")
    print("vocab size", vocab_size, "context length", context_length)

    if time_measure_params is None:
        time_measure_params = {
            "warmup_count": WARMUP_COUNT_BEFORE_TIME,
            "measure_for_count": MEASURE_FOR_COUNT
        }

    if not generate_data_randomly:
        if train_data is None:
            raise Exception("Generate randomly is false, so train data cannot be None")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    if use_mixed_precision:
        print("USING MIXED PRECISION")
        autocast_context = torch.autocast(device_type=device, dtype=torch.bfloat16)
    else:
        autocast_context = nullcontext()

    model = basic_model.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )
    model.to(DEVICE)

#   model = torch.compile(model)
    print("model initialized")

    if not use_homegrown_adam:
        optimizer = basic_optimizer.AdamW(
            model.parameters(),
            lr= min_learning_rate,
            betas = BETAS,
            weight_decay = WEIGHT_DECAY
        )
        print("given adam")
    else:
        optimizer = basic_optimizer.AdamWHomeGrown(
            model.parameters(),
            lr=min_learning_rate,
            betas=BETAS,
            weight_decay=WEIGHT_DECAY
        )
        print("homegrown adam")
    print("optimizer ready")

    it_start = 0

    res = defaultdict(lambda: [])

    ## WARMUP PHASE
    with autocast_context:
        with nvtx.range("WARM_UP"):
            for it_id in range(it_start, time_measure_params['warmup_count']):
                print("it_id", it_id)
                current_lr = basic_optimizer.get_cosine_lr(
                    it=it_id,
                    max_learning_rate=max_learning_rate,
                    min_learning_rate=min_learning_rate,
                    warmup_iters=WARMUP_ITERS,
                    cosine_cycle_iters=ITERATIONS,
                )

                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                # wandb.log({
                #     "learning_rate": float(current_lr)
                # }, step=it_id)

                # input_tensor, target_tensor = data_loader(train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
                if generate_data_randomly:
                    input_tensor, target_tensor = get_random_data(batch_size=batch_size, context_length=context_length,
                                                              device=device)
                else:
                    input_tensor, target_tensor = data_loader(train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)


                optimizer.zero_grad()

                logits = model(input_tensor)

                loss = basic_nn_utils.cross_entropy(logits, target_tensor)

            loss.backward()
            basic_nn_utils.clip_gradient(
                    model.parameters(), max_norm=1.0
                )

            optimizer.step()

    with autocast_context:
        for it_id in range(time_measure_params['measure_for_count']):
            with nvtx.range("FULL_TRAIN_RUN"):
                # print("it_id", it_id, "Counting time")
                current_lr = basic_optimizer.get_cosine_lr(
                    it=it_id,
                    max_learning_rate=max_learning_rate,
                    min_learning_rate=min_learning_rate,
                    warmup_iters=WARMUP_ITERS,
                    cosine_cycle_iters=ITERATIONS,
                )

                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                # wandb.log({
                #     "learning_rate": float(current_lr)
                # }, step=it_id)

                # input_tensor, target_tensor = data_loader(train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
                if generate_data_randomly:
                    input_tensor, target_tensor = get_random_data(batch_size=batch_size, context_length=context_length,
                                                                  device=device)
                else:
                    input_tensor, target_tensor = data_loader(train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)

                optimizer.zero_grad()

                forward_start_time = timeit.default_timer()
                with nvtx.range("FORWARD_PASS"):
                    logits = model(input_tensor)

                conditionally_torch_sync(device)
                res['FORWARD_PASS_TIME'].append(timeit.default_timer() - forward_start_time)

                loss = basic_nn_utils.cross_entropy(logits, target_tensor)

            backward_start_time = timeit.default_timer()
            with nvtx.range("BACKWARD_PASS"):
                loss.backward()

            conditionally_torch_sync(device)
            res['BACKWARD_PASS_TIME'].append(timeit.default_timer() - backward_start_time)

            basic_nn_utils.clip_gradient(
                model.parameters(), max_norm=1.0
            )

            with nvtx.range("OPTIMIZER_STEP"):
                optimizer.step()

    return res



if __name__ == '__main__':
    print("MAIN DRIVER IS RUNNING")
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=VOCAB_SIZE)
    parser.add_argument('--context_length', type=int, default=CONTEXT_LENGTH)
    parser.add_argument('--d_model', type=int, default=D_MODEL)
    parser.add_argument('--num_layers', type=int, default=NUM_BLOCKS)
    parser.add_argument('--num_heads', type=int, default=NUM_HEADS)
    parser.add_argument('--d_ff', type=int, default=D_FF)
    parser.add_argument('--rope_theta', type=float, default=ROPE_THETA)
    parser.add_argument('--max_learning_rate', type=float, default=MAX_LEARNING_RATE)
    parser.add_argument('--min_learning_rate', type=float, default=MIN_LEARNING_RATE)

    args = parser.parse_args()

    res = training_loop(
        max_learning_rate=args.max_learning_rate,
        min_learning_rate=args.min_learning_rate,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=DEVICE
    )
    print(res)