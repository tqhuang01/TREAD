import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from lib import (
    compute_all_losses, evaluation,
    parse_datasets,
    setup_seed, setup_logging, fmt_args, get_next_batch,
)

from model import TREAD, FM_ZP

MODEL_MAP = {
    "TREAD": TREAD,
    "FM_ZP": FM_ZP,
}

BACKBONE_MAP = {
    "MOMENT": "./core/tsfm/AutonLab/MOMENT-1-small",
}


def parse_args():
    parser = argparse.ArgumentParser('IMTS Forecasting')

    # Basic
    parser.add_argument('--state', type=str, default='def')
    parser.add_argument('--seed', type=int)

    # Dataset
    parser.add_argument('--num_samples', '-n', type=int, default=int(1e8))
    parser.add_argument('--dataset', type=str, required=True, choices=["physionet", "mimic", "ushcn", "activity"])
    parser.add_argument('--batch_size', '-b', type=int, required=True)
    parser.add_argument('--history', type=int, required=True,
                        help="number of hours (months for ushcn and ms for activity) as historical window")
    # value 0 means using original time granularity, Value 1 means quantization by 1 hour,
    # value 0.1 means quantization by 0.1 hour = 6 min, value 0.016 means quantization by 0.016 hour = 1 min
    parser.add_argument('--quantization', type=float, default=0.0, choices=[0.0, 1.0, 0.1, 0.016],
                        help="Quantization on the physionet dataset.")
    parser.add_argument('--few_shot_ratio', type=float, default=None, choices=[0.1, 0.2, 0.5], help='train percent')

    # Patch
    parser.add_argument('--patch_size', type=float, choices=[2, 8, 300])
    parser.add_argument('--stride', type=int, choices=[2, 8, 300])
    parser.add_argument('--patch_ts', type=bool, default=False)

    # Training
    parser.add_argument('--epochs', type=int, default=1000, help="Training epochs")
    parser.add_argument('--patience', type=int, default=10, help="Early stopping patience")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")

    # Model
    parser.add_argument('--model_name', type=str, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--backbone", type=str, choices=list(BACKBONE_MAP.keys()))

    parser.add_argument('--ctx_len', type=int, required=True)
    parser.add_argument('--fcast_len', type=int, required=True)
    parser.add_argument('--use_padding', type=str, choices=['none', 'periodic', 'zero'])

    # Transformer
    parser.add_argument('--num_blocks', type=int, default=1, help="TS model layers")
    parser.add_argument('--nhead', type=int, default=1, help="Transformer heads")
    parser.add_argument('--num_layers', type=int, default=1, help="Number of Transformer encoder layers")

    # GNN
    parser.add_argument('-nd', '--node_dim', type=int, default=12, help="Number of units for node vectors")
    parser.add_argument('--num_hops', type=int, default=1, help="hops in GNN")

    # Dimensions
    parser.add_argument('--hidden_dim', '-hd', type=int, default=128, help="Hidden size")
    parser.add_argument('--te_dim', '-td', type=int, default=12, help="Time encoding size")

    # Hardware
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--pid', type=int, default=os.getpid())

    args = parser.parse_args()

    # Extra params
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.patch_ts:
        args.num_patches = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1  # (window size for a patch)

    return args


def build_model(args):
    args.model_dir = BACKBONE_MAP.get(args.backbone)

    return MODEL_MAP[args.model_name](args).to(args.device)


def main():
    args = parse_args()

    # Set seed
    setup_seed(args.seed)

    # Load dataset
    data_obj = parse_datasets(args, args.patch_ts)
    args.num_features = data_obj["input_dim"]

    # Setup logging
    padding_type = args.use_padding.capitalize()  # "Circular" or "Zero" or "None"
    log_dataset_name = f"{args.dataset}/{args.backbone}_{padding_type}_{args.ctx_len}-{args.fcast_len}_{args.batch_size}bs_{args.num_blocks}b_{args.nhead}h_{args.num_layers}l_{args.te_dim}td_{args.node_dim}nd_{args.num_hops}k_{args.hidden_dim}hd"
    if args.few_shot_ratio is not None:
        log_dataset_name = f"{args.dataset}_{args.few_shot_ratio}/{args.backbone}_{padding_type}_{args.ctx_len}-{args.fcast_len}_{args.batch_size}bs_{args.num_blocks}b_{args.nhead}h_{args.num_layers}l_{args.te_dim}td_{args.node_dim}nd_{args.num_hops}k_{args.hidden_dim}hd"

    logger = setup_logging(
        logger_name=args.model_name,
        base_log_dir="logs",
        model_name=args.model_name,
        dataset_name=log_dataset_name,
    )

    logger.info(" ".join(sys.argv))
    logger.info(args)
    logger.info(f"\n{fmt_args(args)}")
    logger.info(f"PID={args.pid}, Device={args.device}.")
    logger.info(f"input_dim: {args.num_features}")
    logger.info(f"n_train_batches: {data_obj['n_train_batches']}")
    logger.info(f"n_val_batches: {data_obj['n_val_batches']}")
    logger.info(f"n_test_batches: {data_obj['n_test_batches']}")

    model = build_model(args)

    optimizer = optim.Adam(
        # model.parameters(),
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_mse = np.inf
    best_iter = -1
    test_res = {}
    for itr in range(args.epochs):
        start_time = time.time()

        # Training
        model.train()
        for _ in tqdm(range(data_obj["n_train_batches"])):
            optimizer.zero_grad()
            batch = get_next_batch(data_obj["train_dataloader"])
            train_res = compute_all_losses(model, batch)
            train_res["loss"].backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])

            # Check if validation improves
            if val_res["mse"] < best_val_mse:
                best_val_mse = val_res["mse"]
                best_iter = itr

                # Testing
                test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])

        # Logging
        logger.info(
            f"Epoch {itr:03d}(Best @ {best_iter:03d}) - "
            f"Train: MSE={train_res['mse']:.5f} - "
            f"Val: MSE={val_res['mse']:.5f}, MAE={val_res['mae']:.5f} - "
            f"Test: MSE={test_res['mse']:.5f}, MAE={test_res['mae']:.5f} - "
            f"Time Spent: {time.time() - start_time:.2f}s"
        )

        # Early stopping
        if itr - best_iter >= args.patience:
            logger.info("Exp has been early stopped!")
            sys.exit(0)


if __name__ == '__main__':
    main()
