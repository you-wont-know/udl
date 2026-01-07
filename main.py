from data import get_mnist
from cnn import CNN, AnalyticalCNN, VariationalCNN
from episode import episode, evaluate_pool_scores, evaluate_pool_scores_two_stage
from acquisition_functions import *

import time
import torch
import argparse

# These are not going to change, so I just define them here
INITIAL_TRAIN_SIZE = 20
VAL_SIZE = 100
POOL_SIZE = 60000 - INITIAL_TRAIN_SIZE - VAL_SIZE
TEST_SIZE = 10000

BATCH_SIZE = 32
EPISODES = 100

DEFAULT_T = 35
DETERMINISTIC_T = 1

# Hyperparameters to try out for the extension models
PRIOR_S_VALUES = [0.1, 0.5, 1.0, 2.0]


ACQ_FUNCTIONS = {
    "random_acquisition": random_acquisition,
    "bald_acquisition": bald_acquisition,
    "predictive_entropy_acquisition": predictive_entropy_acquisition,
    "variation_ratio_acquisition": variation_ratio_acquisition,
    "mean_standard_deviation_acquisition": mean_standard_deviation_acquisition,
    "determinant_acquisition": determinant_acquisition,
    "bayesian_entropy_acquisition": bayesian_entropy_acquisition,
    "determinant_entropy_acquisition": determinant_entropy_acquisition,
    "max_diag_acquisition": max_diag_acquisition,
    "trace_acquisition": trace_acquisition,
    "bayesian_variation_ratio_acquisition": bayesian_variation_ratio_acquisition,
}

POOL_SCORE_FUNCTIONS = {
    "evaluate_pool_scores": evaluate_pool_scores,
    "evaluate_pool_scores_two_stage": evaluate_pool_scores_two_stage,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--acq", choices=sorted(ACQ_FUNCTIONS.keys()), required=True)
    parser.add_argument("--pool-scores", choices=sorted(POOL_SCORE_FUNCTIONS.keys()), required=True)
    parser.add_argument("--model", choices=["cnn", "analytical", "variational"], default="cnn")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()

args = parse_args()
acq_function = ACQ_FUNCTIONS[args.acq]
evaluate_pool_scores_fn = POOL_SCORE_FUNCTIONS[args.pool_scores]

if args.model != "cnn":
    dropout_samples = DETERMINISTIC_T
    acquisition_T = DETERMINISTIC_T
    use_dropout = False
    mode_name = args.model + "-" + args.acq
    BATCH_SIZE = 1024 + 512 # single-batch training and evaluation
elif args.deterministic:
    dropout_samples = DETERMINISTIC_T
    acquisition_T = DETERMINISTIC_T
    use_dropout = False # do NOT use dropout DURING TESTING
    mode_name = "deterministic"
else:
    dropout_samples = DEFAULT_T
    acquisition_T = DEFAULT_T
    use_dropout = True  # do use dropout DURING TESTING
    mode_name = "main"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CLASSES = {
    "cnn": CNN,
    "analytical": AnalyticalCNN,
    "variational": VariationalCNN,
}


LOG_PATH = f"logs/{mode_name}-{args.acq}-{args.seed}.log"

with open(LOG_PATH, "w") as log_file:
    log_file.write(f"seed: {args.seed}\n")
    log_file.write(f"function: {acq_function.__name__}\n")
    log_file.write(f"pool_scores_fn: {evaluate_pool_scores_fn.__name__}\n")
    log_file.write(f"mode: {mode_name}\n")
    log_file.write(f"use_dropout: {use_dropout}\n")
    log_file.write(f"dropout_samples: {dropout_samples}\n")
    log_file.write(f"acquisition T: {acquisition_T}\n")
    log_file.write(f"model: {args.model}\n")
    log_file.write(f"sigma: {args.sigma}\n")
    log_file.write(f"\n")
    log_file.flush()

    train, val, pool, test = get_mnist(INITIAL_TRAIN_SIZE, VAL_SIZE, TEST_SIZE, POOL_SIZE, device)

    print("======")
    print("== Acquisition function: ", acq_function.__name__, ". Mode", mode_name, "\n")
    for i in range(EPISODES):
        print(f"Episode {i+1}")
        start_time = time.time()

        model_class = MODEL_CLASSES[args.model]
        model_params = {}
        s_candidates = None
        if args.model == "analytical":
            sigma_epsilon = (args.sigma ** 2) * torch.eye(10, device=device)
            model_params = {"sigma_epsilon": sigma_epsilon, "k": 128}
            s_candidates = PRIOR_S_VALUES
        elif args.model == "variational":
            model_params = {"sigma": args.sigma, "k": 128}
            s_candidates = PRIOR_S_VALUES

        train, val, pool, test, test_acc, test_rmse = episode(
            model_class,
            train,
            val,
            pool,
            test,
            acq_function,
            evaluate_pool_scores_fn,
            BATCH_SIZE,
            device,
            dropout_samples=dropout_samples,
            acquisition_T=acquisition_T,
            use_dropout=use_dropout,
            model_kind=args.model,
            model_params=model_params,
            s_candidates=s_candidates,
            k=10,
        )
        end_time = time.time()
        print("Test accuracy", test_acc)
        if test_rmse is not None:
            print("Test RMSE", test_rmse)
        print("This episode took", (end_time-start_time), "seconds", flush=True)
        print()
        if test_rmse is None:
            log_file.write(f"episode: {i+1} accuracy: {test_acc}\n")
        else:
            log_file.write(f"episode: {i+1} accuracy: {test_acc} rmse: {test_rmse}\n")
        log_file.flush()
