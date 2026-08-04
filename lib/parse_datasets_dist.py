import math

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, DistributedSampler

import lib.utils as utils
from lib.mimic import MIMIC
from lib.person_activity import *
from lib.physionet import *
from lib.ushcn import *


def parse_datasets_dist(args, patch_ts=False, length_stat=False, rank=None, world_size=None):
    device = args.device
    dataset_name = args.dataset

    if world_size is None:
        world_size = getattr(args, "world_size", 1)
    distributed = rank is not None and world_size > 1
    data_device = torch.device("cpu") if distributed else device

    total_dataset = None
    train_data = None
    val_data = None
    test_data = None

    if dataset_name in ["physionet", "mimic"]:
        if dataset_name == "physionet":
            total_dataset = PhysioNet(
                "./data/physionet",
                download=True,
                quantization=args.quantization,
                n_samples=args.num_samples,
                device=data_device,
            )
        elif dataset_name == "mimic":
            total_dataset = MIMIC(
                "./data/mimic/",
                n_samples=args.num_samples,
                device=data_device,
            )

        seen_data, test_data = train_test_split(total_dataset, train_size=0.8, random_state=42, shuffle=True)
        train_data, val_data = train_test_split(seen_data, train_size=0.75, random_state=42, shuffle=False)

        if args.few_shot_ratio is not None:
            train_data, trash = train_test_split(train_data, train_size=args.few_shot_ratio, random_state=42,
                                                 shuffle=False)

        _, _, vals, _ = train_data[0]
        input_dim = vals.size(-1)

        global_batch_size = min(min(len(seen_data), args.batch_size), args.num_samples)
        batch_size = max(1, math.ceil(global_batch_size / world_size)) if distributed else global_batch_size
        data_min, data_max, time_max = get_data_min_max(seen_data, data_device)

        base_collate_fn = patch_variable_time_collate_fn if patch_ts else variable_time_collate_fn

        train_sampler = DistributedSampler(
            train_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        ) if distributed else None

        val_data_rank = val_data[rank::world_size] if distributed else val_data
        test_data_rank = test_data[rank::world_size] if distributed else test_data

        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=lambda batch: base_collate_fn(
                batch,
                args,
                data_device,
                data_type="train",
                data_min=data_min,
                data_max=data_max,
                time_max=time_max,
            ),
        )
        val_dataloader = DataLoader(
            val_data_rank,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: base_collate_fn(
                batch,
                args,
                data_device,
                data_type="val",
                data_min=data_min,
                data_max=data_max,
                time_max=time_max,
            ),
        )
        test_dataloader = DataLoader(
            test_data_rank,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: base_collate_fn(
                batch,
                args,
                data_device,
                data_type="test",
                data_min=data_min,
                data_max=data_max,
                time_max=time_max,
            ),
        )

        data_objects = {
            "train_dataloader": utils.inf_generator(train_dataloader),
            "val_dataloader": utils.inf_generator(val_dataloader),
            "test_dataloader": utils.inf_generator(test_dataloader),
            "train_sampler": train_sampler,
            "val_sampler": None,
            "test_sampler": None,
            "input_dim": input_dim,
            "n_train_batches": len(train_dataloader),
            "n_val_batches": len(val_dataloader),
            "n_test_batches": len(test_dataloader),
            "data_max": data_max,
            "data_min": data_min,
            "time_max": time_max,
            "global_batch_size": global_batch_size,
            "batch_size": batch_size,
            "distributed": distributed,
        }

        if length_stat:
            max_input_len, max_pred_len, median_len = get_seq_length(args, total_dataset)
            data_objects["max_input_len"] = max_input_len.item()
            data_objects["max_pred_len"] = max_pred_len.item()
            data_objects["median_len"] = median_len.item()
            if rank in (None, 0):
                print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

    elif dataset_name == "ushcn":
        args.n_months = 48
        args.pred_window = 1

        total_dataset = USHCN("./data/ushcn/", n_samples=args.num_samples, device=data_device)

        seen_data, test_data = train_test_split(total_dataset, train_size=0.8, random_state=42, shuffle=True)
        train_data, val_data = train_test_split(seen_data, train_size=0.75, random_state=42, shuffle=False)

        if args.few_shot_ratio is not None:
            train_data, trash = train_test_split(train_data, train_size=args.few_shot_ratio, random_state=42,
                                                 shuffle=False)

        _, _, vals, _ = train_data[0]
        input_dim = vals.size(-1)

        data_min, data_max, time_max = get_data_min_max(seen_data, data_device)
        base_collate_fn = USHCN_patch_variable_time_collate_fn if patch_ts else USHCN_variable_time_collate_fn

        train_data = USHCN_time_chunk(train_data, args, data_device)
        val_data = USHCN_time_chunk(val_data, args, data_device)
        test_data = USHCN_time_chunk(test_data, args, data_device)

        global_batch_size = args.batch_size
        batch_size = max(1, math.ceil(global_batch_size / world_size)) if distributed else global_batch_size

        train_sampler = DistributedSampler(
            train_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        ) if distributed else None

        val_data_rank = val_data[rank::world_size] if distributed else val_data
        test_data_rank = test_data[rank::world_size] if distributed else test_data

        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=lambda batch: base_collate_fn(batch, args, data_device, time_max=time_max),
        )
        val_dataloader = DataLoader(
            val_data_rank,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: base_collate_fn(batch, args, data_device, time_max=time_max),
        )
        test_dataloader = DataLoader(
            test_data_rank,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: base_collate_fn(batch, args, data_device, time_max=time_max),
        )

        data_objects = {
            "train_dataloader": utils.inf_generator(train_dataloader),
            "val_dataloader": utils.inf_generator(val_dataloader),
            "test_dataloader": utils.inf_generator(test_dataloader),
            "train_sampler": train_sampler,
            "val_sampler": None,
            "test_sampler": None,
            "input_dim": input_dim,
            "n_train_batches": len(train_dataloader),
            "n_val_batches": len(val_dataloader),
            "n_test_batches": len(test_dataloader),
            "data_max": data_max,
            "data_min": data_min,
            "time_max": time_max,
            "global_batch_size": global_batch_size,
            "batch_size": batch_size,
            "distributed": distributed,
        }

        if length_stat:
            max_input_len, max_pred_len, median_len = USHCN_get_seq_length(args, train_data + val_data + test_data)
            data_objects["max_input_len"] = max_input_len.item()
            data_objects["max_pred_len"] = max_pred_len.item()
            data_objects["median_len"] = median_len.item()
            if rank in (None, 0):
                print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

    elif dataset_name == "activity":
        args.pred_window = 1000

        total_dataset = PersonActivity(
            "./data/activity/",
            download=True,
            n_samples=args.num_samples,
            device=data_device,
        )

        seen_data, test_data = train_test_split(total_dataset, train_size=0.8, random_state=42, shuffle=True)
        train_data, val_data = train_test_split(seen_data, train_size=0.75, random_state=42, shuffle=False)

        if args.few_shot_ratio is not None:
            train_data, trash = train_test_split(train_data, train_size=args.few_shot_ratio, random_state=42,
                                                 shuffle=False)

        _, _, vals, _ = train_data[0]
        input_dim = vals.size(-1)

        global_batch_size = min(min(len(seen_data), args.batch_size), args.num_samples)
        batch_size = max(1, math.ceil(global_batch_size / world_size)) if distributed else global_batch_size
        data_min, data_max, _ = get_data_min_max(seen_data, data_device)
        time_max = torch.tensor(args.history + args.pred_window, device=data_device)
        if rank in (None, 0):
            print("manual set time_max:", time_max)

        base_collate_fn = patch_variable_time_collate_fn if patch_ts else variable_time_collate_fn

        train_data = Activity_time_chunk(train_data, args, data_device)
        val_data = Activity_time_chunk(val_data, args, data_device)
        test_data = Activity_time_chunk(test_data, args, data_device)

        train_sampler = DistributedSampler(
            train_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        ) if distributed else None

        val_data_rank = val_data[rank::world_size] if distributed else val_data
        test_data_rank = test_data[rank::world_size] if distributed else test_data

        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=lambda batch: base_collate_fn(
                batch,
                args,
                data_device,
                data_type="train",
                data_min=data_min,
                data_max=data_max,
                time_max=time_max,
            ),
        )
        val_dataloader = DataLoader(
            val_data_rank,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: base_collate_fn(
                batch,
                args,
                data_device,
                data_type="val",
                data_min=data_min,
                data_max=data_max,
                time_max=time_max,
            ),
        )
        test_dataloader = DataLoader(
            test_data_rank,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: base_collate_fn(
                batch,
                args,
                data_device,
                data_type="test",
                data_min=data_min,
                data_max=data_max,
                time_max=time_max,
            ),
        )

        data_objects = {
            "train_dataloader": utils.inf_generator(train_dataloader),
            "val_dataloader": utils.inf_generator(val_dataloader),
            "test_dataloader": utils.inf_generator(test_dataloader),
            "train_sampler": train_sampler,
            "val_sampler": None,
            "test_sampler": None,
            "input_dim": input_dim,
            "n_train_batches": len(train_dataloader),
            "n_val_batches": len(val_dataloader),
            "n_test_batches": len(test_dataloader),
            "data_max": data_max,
            "data_min": data_min,
            "time_max": time_max,
            "global_batch_size": global_batch_size,
            "batch_size": batch_size,
            "distributed": distributed,
        }

        if length_stat:
            max_input_len, max_pred_len, median_len = Activity_get_seq_length(args, train_data + val_data + test_data)
            data_objects["max_input_len"] = max_input_len.item()
            data_objects["max_pred_len"] = max_pred_len.item()
            data_objects["median_len"] = median_len.item()
            if rank in (None, 0):
                print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if rank in (None, 0):
        print("+-----------------------+")
        print("| Dataset Split Summary |")
        print("+-----------------------+")
        print(f"| {'Total':<6}: {len(total_dataset):<13} |")
        print(f"| {'Train':<6}: {len(train_data):<13} |")
        print(f"| {'Val':<6}: {len(val_data):<13} |")
        print(f"| {'Test':<6}: {len(test_data):<13} |")
        print("+-----------------------+")

    return data_objects
