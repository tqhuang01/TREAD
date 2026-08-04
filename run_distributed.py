import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from lib import (
    compute_all_losses, compute_error, evaluation,
    parse_datasets_dist,
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


def setup(rank, world_size, master_port):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.barrier()


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser("IMTS Forecasting")

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
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")

    # Model
    parser.add_argument('--model_name', type=str, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--backbone", type=str, choices=list(BACKBONE_MAP.keys()))

    parser.add_argument("--ctx_len", type=int)
    parser.add_argument("--fcast_len", type=int)
    parser.add_argument('--use_padding', type=str, choices=['none', 'periodic', 'zero'])

    # Transformer
    parser.add_argument("--num_blocks", type=int, default=1, help="TS model layers")
    parser.add_argument("--nhead", type=int, default=1, help="Transformer heads")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of Transformer encoder layers")

    # GNN
    parser.add_argument("-nd", "--node_dim", type=int, default=12, help="Number of units for node vectors")
    parser.add_argument("--num_hops", type=int, default=1, help="hops in GNN")

    # Dimensions
    parser.add_argument("--hidden_dim", "-hd", type=int, default=128, help="Hidden size")
    parser.add_argument("--te_dim", "-td", type=int, default=12, help="Time encoding size")

    # Hardware
    parser.add_argument('--gpu', type=str, default='0,1', help='GPU id')
    parser.add_argument("--master_port", type=int, default=29500, help="DDP master port")

    args = parser.parse_args()

    # Extra params
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    return args


def build_model(args):
    args.model_dir = BACKBONE_MAP[args.backbone]

    return MODEL_MAP[args.model_name](args).to(args.device)


def reduce_train_results(train_res, device, world_size):
    values = torch.tensor(
        [
            train_res["loss"].detach().item(),
            train_res["mse"],
            train_res["mae"],
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= world_size

    mse = values[1].item()
    return {
        "loss": values[0].item(),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": values[2].item(),
    }


def evaluation_distributed(model, dataloader, n_batches, device, n_features, disable_tqdm=False):
    total_se = None
    total_ae = None
    total_ape = None
    total_mask = None
    total_mask_mape = None

    for _ in tqdm(range(n_batches), disable=disable_tqdm):
        batch_dict = get_next_batch(dataloader)
        batch_dict = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch_dict.items()
        }

        pred_y = model(
            batch_dict["tp_to_predict"],
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            batch_dict["observed_mask"],
        )

        se_var_sum, mask_count = compute_error(
            batch_dict["data_to_predict"],
            pred_y,
            mask=batch_dict["mask_predicted_data"],
            func="MSE",
            reduce="sum",
        )
        ae_var_sum, _ = compute_error(
            batch_dict["data_to_predict"],
            pred_y,
            mask=batch_dict["mask_predicted_data"],
            func="MAE",
            reduce="sum",
        )
        ape_var_sum, mask_count_mape = compute_error(
            batch_dict["data_to_predict"],
            pred_y,
            mask=batch_dict["mask_predicted_data"],
            func="MAPE",
            reduce="sum",
        )

        if total_se is None:
            total_se = torch.zeros_like(se_var_sum)
            total_ae = torch.zeros_like(ae_var_sum)
            total_ape = torch.zeros_like(ape_var_sum)
            total_mask = torch.zeros_like(mask_count)
            total_mask_mape = torch.zeros_like(mask_count_mape)

        total_se += se_var_sum
        total_ae += ae_var_sum
        total_ape += ape_var_sum
        total_mask += mask_count
        total_mask_mape += mask_count_mape

    if total_se is None:
        total_se = torch.zeros(n_features, dtype=torch.float32, device=device)
        total_ae = torch.zeros(n_features, dtype=torch.float32, device=device)
        total_ape = torch.zeros(n_features, dtype=torch.float32, device=device)
        total_mask = torch.zeros(n_features, dtype=torch.float32, device=device)
        total_mask_mape = torch.zeros(n_features, dtype=torch.float32, device=device)

    for value in (total_se, total_ae, total_ape, total_mask, total_mask_mape):
        dist.all_reduce(value, op=dist.ReduceOp.SUM)

    n_avai_var = torch.count_nonzero(total_mask)
    n_avai_var_mape = torch.count_nonzero(total_mask_mape)

    mse = (total_se / (total_mask + 1e-8)).sum() / n_avai_var
    mae = (total_ae / (total_mask + 1e-8)).sum() / n_avai_var
    mape = (total_ape / (total_mask_mape + 1e-8)).sum() / n_avai_var_mape

    return {
        "loss": mse.item(),
        "mse": mse.item(),
        "mae": mae.item(),
        "rmse": torch.sqrt(mse).item(),
        "mape": mape.item(),
    }


def main(args):
    setup_seed(args.seed)

    data_obj = parse_datasets_dist(args)
    args.num_features = data_obj["input_dim"]

    padding_type = args.use_padding.capitalize()
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
    logger.info(f"PID={os.getpid()}, Device={args.device}.")
    logger.info(f"input_dim: {args.num_features}")
    logger.info(f"global_batch_size: {data_obj['global_batch_size']}")
    logger.info(f"per_process_batch_size: {data_obj['batch_size']}")
    logger.info(f"n_train_batches: {data_obj['n_train_batches']}")
    logger.info(f"n_val_batches: {data_obj['n_val_batches']}")
    logger.info(f"n_test_batches: {data_obj['n_test_batches']}")

    model = build_model(args)
    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_mse = np.inf
    best_iter = -1
    test_res = {}
    for itr in range(args.epochs):
        start_time = time.time()

        model.train()
        for _ in tqdm(range(data_obj["n_train_batches"])):
            optimizer.zero_grad(set_to_none=True)
            batch = get_next_batch(data_obj["train_dataloader"])
            batch = {
                key: value.to(args.device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            train_res = compute_all_losses(model, batch)
            train_res["loss"].backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
            if val_res["mse"] < best_val_mse:
                best_val_mse = val_res["mse"]
                best_iter = itr
                test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])

        logger.info(
            f"Epoch {itr:03d}(Best @ {best_iter:03d}) - "
            f"Train: MSE={train_res['mse']:.5f} - "
            f"Val: MSE={val_res['mse']:.5f}, MAE={val_res['mae']:.5f} - "
            f"Test: MSE={test_res['mse']:.5f}, MAE={test_res['mae']:.5f} - "
            f"Time Spent: {time.time() - start_time:.2f}s"
        )

        if itr - best_iter >= args.patience:
            logger.info("Exp has been early stopped!")
            break


def ddp_main(rank, args):
    setup(rank, args.world_size, args.master_port)
    try:
        setup_seed(args.seed + rank)
        args.device = torch.device(f"cuda:{rank}")

        data_obj = parse_datasets_dist(args, rank=rank, world_size=args.world_size)
        args.num_features = data_obj["input_dim"]

        logger = None
        if rank == 0:
            padding_type = args.use_padding.capitalize()
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
            logger.info(f"PID={os.getpid()}, Device={args.device}.")
            logger.info(f"input_dim: {args.num_features}")
            logger.info(f"global_batch_size: {data_obj['global_batch_size']}")
            logger.info(f"per_process_batch_size: {data_obj['batch_size']}")
            logger.info(f"n_train_batches: {data_obj['n_train_batches']}")
            logger.info(f"n_val_batches: {data_obj['n_val_batches']}")
            logger.info(f"n_test_batches: {data_obj['n_test_batches']}")

        # MOMENT model loading rewrites config.json; serialize construction to avoid races.
        model = None
        for build_rank in range(args.world_size):
            if rank == build_rank:
                model = build_model(args)
            dist.barrier()

        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )

        optimizer = optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_val_mse = np.inf
        best_iter = -1
        test_res = {"mse": np.inf, "mae": np.inf}

        for itr in range(args.epochs):
            if data_obj["train_sampler"] is not None:
                data_obj["train_sampler"].set_epoch(itr)

            start_time = time.time()

            model.train()
            for _ in tqdm(range(data_obj["n_train_batches"]), disable=rank != 0):
                optimizer.zero_grad(set_to_none=True)
                batch = get_next_batch(data_obj["train_dataloader"])
                batch = {
                    key: value.to(args.device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                train_res = compute_all_losses(model, batch)
                train_res["loss"].backward()
                optimizer.step()

            train_log = reduce_train_results(train_res, args.device, args.world_size)

            model.eval()
            with torch.no_grad():
                val_res = evaluation_distributed(
                    model,
                    data_obj["val_dataloader"],
                    data_obj["n_val_batches"],
                    args.device,
                    args.num_features,
                    disable_tqdm=rank != 0,
                )

                if val_res["mse"] < best_val_mse:
                    best_val_mse = val_res["mse"]
                    best_iter = itr
                    test_res = evaluation_distributed(
                        model,
                        data_obj["test_dataloader"],
                        data_obj["n_test_batches"],
                        args.device,
                        args.num_features,
                        disable_tqdm=rank != 0,
                    )

            if rank == 0:
                logger.info(
                    f"Epoch {itr:03d}(Best @ {best_iter:03d}) - "
                    f"Train: MSE={train_log['mse']:.5f} - "
                    f"Val: MSE={val_res['mse']:.5f}, MAE={val_res['mae']:.5f} - "
                    f"Test: MSE={test_res['mse']:.5f}, MAE={test_res['mae']:.5f} - "
                    f"Time Spent: {time.time() - start_time:.2f}s"
                )

            if itr - best_iter >= args.patience:
                if rank == 0:
                    logger.info("Exp has been early stopped!")
                break
    finally:
        cleanup()


if __name__ == "__main__":
    args = parse_args()
    if args.world_size <= 1:
        main(args)
    else:
        mp.spawn(ddp_main, args=(args,), nprocs=args.world_size, join=True)
