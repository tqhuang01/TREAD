import os
import logging

import torch

import numpy as np

import datetime
import random


def setup_seed(seed, reproduce: bool = True):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if reproduce:  # Set seeds for reproducibility
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def setup_logging(
        logger_name: str,
        base_log_dir: str,
        model_name: str,
        dataset_name: str,
        log_level: int = logging.INFO,
):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    if logger.handlers:  # Clear existing handlers
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Create log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_log_dir = os.path.join(base_log_dir, f"{model_name}/{dataset_name}")
    os.makedirs(full_log_dir, exist_ok=True)
    log_file_path = os.path.join(full_log_dir, f"run_{timestamp}.log")

    # Set format for both handlers
    formatter = logging.Formatter(
        "{asctime} | {levelname} | {name} | {filename:<15}:{lineno:<4} | {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style='{'
    )

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


def get_next_batch(dataloader):
    # Make the union of all time points and perform normalization across the whole dataset
    data_dict = dataloader.__next__()

    return data_dict


def normalize_masked_data(data, mask, att_min, att_max):
    scale = att_max - att_min
    scale = scale + (scale == 0) * 1e-8
    # we don't want to divide by zero
    if (scale != 0.).all():
        data_norm = (data - att_min) / scale
    else:
        raise Exception("Zero!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    return data_norm


def normalize_masked_tp(data, att_min, att_max):
    scale = att_max - att_min
    scale = scale + (scale == 0) * 1e-8
    # we don't want to divide by zero
    if (scale != 0.).all():
        data_norm = (data - att_min) / scale
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    return data_norm


def split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices):
    device = get_device(data_dict["data"])

    split_dict = {"tp_to_predict": data_dict["tp_to_predict"].clone(),
                  "data_to_predict": data_dict["data_to_predict"].clone(),
                  "mask_predicted_data": data_dict["mask_predicted_data"].clone()
                  }

    observed_tp = data_dict["time_steps"].clone()  # (n_observed_tp, )
    observed_data = data_dict["data"].clone()  # (bs, n_observed_tp, D)
    observed_mask = data_dict["mask"].clone()  # (bs, n_observed_tp, D)

    n_batch, n_tp, n_dim = observed_data.shape
    observed_tp_patches = observed_tp.view(1, 1, -1, 1).repeat(n_batch, args.num_patches, 1, n_dim)
    observed_data_patches = observed_data.view(n_batch, 1, n_tp, n_dim).repeat(1, args.num_patches, 1, 1)
    observed_mask_patches = observed_mask.view(n_batch, 1, n_tp, n_dim).repeat(1, args.num_patches, 1, 1)

    max_patch_len = 0
    for i in range(args.num_patches):
        indices = patch_indices[i]
        if (len(indices) == 0): continue
        st_ind, ed_ind = indices[0], indices[-1]
        n_data_points = observed_mask[:, st_ind:ed_ind + 1].sum(dim=1).max().item()
        max_patch_len = max(max_patch_len, int(n_data_points))

    observed_mask_patches_fill = torch.zeros_like(observed_mask_patches,
                                                  dtype=observed_mask.dtype)  # n_batch, npacth, n_tp, n_dim
    patch_indices_fianl = torch.full((n_batch, args.num_patches, max_patch_len, n_dim), n_tp).to(
        device)  # n_batch, npacth, max_patch_len, n_dim
    observed_mask_patches_fill_reindex = torch.zeros_like(patch_indices_fianl, dtype=observed_mask.dtype)
    aux_tensor = torch.arange(max_patch_len).view(1, max_patch_len, 1).repeat(n_batch, 1, n_dim).to(device)
    for i in range(args.num_patches):
        indices = patch_indices[i]
        if (len(indices) == 0): continue
        st_ind, ed_ind = indices[0], indices[-1]
        observed_mask_patches_fill[:, i, st_ind:ed_ind + 1] = observed_mask[:, st_ind:ed_ind + 1, :]
        L = observed_mask[:, st_ind:ed_ind + 1, :].sum(dim=1, keepdim=True)  # (bs, 1, D)
        observed_mask_patches_fill_reindex[:, i] = (aux_tensor < L)  # let first L[i] to be True

    ### return a indices tuple like ([...], [...], [...], [...])
    mask_inds = torch.nonzero(observed_mask_patches_fill_reindex.permute(0, 1, 3, 2), as_tuple=True)  # reset indices
    ind_values = torch.nonzero(observed_mask_patches_fill.permute(0, 1, 3, 2), as_tuple=True)[
        -1]  # original indices of dimension 2

    ### fill n_tp if the number of observed points are less than max_patch_len
    patch_indices_fianl.index_put_((mask_inds[0], mask_inds[1], mask_inds[3], mask_inds[2]), ind_values)

    pad_zeros_data = torch.zeros([n_batch, args.num_patches, 1, n_dim]).to(device)
    observed_tp_patches = torch.cat([observed_tp_patches, pad_zeros_data], dim=2).gather(2,
                                                                                         patch_indices_fianl)  # (n_batch, npatch, max_patch_len, n_dim)
    observed_data_patches = torch.cat([observed_data_patches, pad_zeros_data], dim=2).gather(2, patch_indices_fianl)
    observed_mask_patches = torch.cat([observed_mask_patches, pad_zeros_data], dim=2).gather(2, patch_indices_fianl)

    split_dict["observed_tp"] = observed_tp_patches
    split_dict["observed_data"] = observed_data_patches
    split_dict["observed_mask"] = observed_mask_patches

    return split_dict


def check_mask(data, mask):
    # check that "mask" argument indeed contains a mask for data
    n_zeros = torch.sum(mask == 0.).cpu().numpy()
    n_ones = torch.sum(mask == 1.).cpu().numpy()

    # mask should contain only zeros and ones
    assert ((n_zeros + n_ones) == np.prod(list(mask.size())))

    # all masked out elements should be zeros
    assert (torch.sum(data[mask == 0.] != 0.) == 0)


def periodic_extension(source, tgt_len: int):
    """
    source: [B, L, D] or [B, L]
    return:[B, tgt_len, D] or [B, tgt_len]
    """
    src_len = source.shape[1]
    if src_len > tgt_len:
        print(f"Cut!  {src_len}->{tgt_len}")
        return source[:, :tgt_len]

    indices = torch.arange(tgt_len, device=source.device) % src_len
    return source.index_select(dim=1, index=indices)


def zero_padding(source, tgt_len: int):
    """
    source: [B, L, D] or [B, L]
    return:[B, tgt_len, D] or [B, tgt_len]
    """
    src_len = source.shape[1]
    if src_len > tgt_len:
        print(f"Cut!  {src_len}->{tgt_len}")
        return source[:, :tgt_len]

    out_shape = list(source.shape)
    out_shape[1] = tgt_len

    out = source.new_zeros(out_shape)
    out[:, :src_len].copy_(source)

    return out


def fmt_args(args, with_idx: bool = True) -> str:
    args_dict = vars(args)
    rows = list(args_dict.items())

    kw = max(len(k) for k, _ in rows)
    vw = max(len(str(v)) for _, v in rows)
    tw = max(len(type(v).__name__) for _, v in rows)

    if with_idx:
        iw = len(str(len(rows)))
        sep = "+" + "-" * (iw + 2) + "+" + "-" * (kw + 2) + "+" + "-" * (vw + 2) + "+" + "-" * (tw + 2) + "+"
        head = f"| {'#'.ljust(iw)} | {'Arg'.ljust(kw)} | {'Value'.ljust(vw)} | {'Type'.ljust(tw)} |"
    else:
        sep = "+" + "-" * (kw + 2) + "+" + "-" * (vw + 2) + "+" + "-" * (tw + 2) + "+"
        head = f"| {'Arg'.ljust(kw)} | {'Value'.ljust(vw)} | {'Type'.ljust(tw)} |"

    lines = [sep, head, sep]

    for i, (k, v) in enumerate(rows, 1):
        type_str = type(v).__name__
        if with_idx:
            lines.append(f"| {str(i).ljust(iw)} | {k.ljust(kw)} | {str(v).ljust(vw)} | {type_str.ljust(tw)} |")
        else:
            lines.append(f"| {k.ljust(kw)} | {str(v).ljust(vw)} | {type_str.ljust(tw)} |")

    lines.append(sep)
    return "\n".join(lines)
