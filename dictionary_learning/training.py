"""
Training dictionaries
"""

import json
import os
from collections import OrderedDict
from contextlib import nullcontext
from queue import Empty
from typing import Optional

import torch as t
import torch.multiprocessing as mp
import wandb
from tqdm import tqdm


def new_wandb_process(config, log_queue, entity, project):
    wandb.init(entity=entity, project=project, config=config, name=config["wandb_name"])
    while True:
        try:
            log = log_queue.get(timeout=1)
            if log == "DONE":
                break
            wandb.log(log)
        except Empty:
            continue
    wandb.finish()


def log_stats(
    trainers: dict,
    step: int,
    act: t.Tensor,
    activations_split_by_head: bool,
    transcoder: bool,
    layer_indices: list[int],
    log_queues: dict = {},
    verbose: bool = False,
):
    with t.no_grad():
        # quick hack to make sure all trainers get the same x
        z = act.clone()
        for i, (name, trainer) in enumerate(trainers.items()):
            log = {}
            act = z.clone()
            relative_idx = layer_indices.index(trainer.config["layer"])

            act = act[:, relative_idx]
            if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
                act = act[..., i, :]
            if not transcoder:
                act, act_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()
                # fraction of variance explained
                total_variance = t.var(act, dim=0).sum()
                residual_variance = t.var(act - act_hat, dim=0).sum()
                frac_variance_explained = 1 - residual_variance / total_variance
                log["frac_variance_explained"] = frac_variance_explained.item()
            else:  # transcoder
                x, x_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()

            if verbose:
                print(
                    f"Step {step}: L0 = {l0}, frac_variance_explained = {frac_variance_explained}"
                )

            # log parameters from training
            log.update(
                {
                    f"{k}": v.cpu().item() if isinstance(v, t.Tensor) else v
                    for k, v in losslog.items()
                }
            )
            log["l0"] = l0
            trainer_log = trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                if isinstance(value, t.Tensor):
                    value = value.cpu().item()
                log[f"{name}"] = value

            if log_queues:
                log_queues[name].put(log)


def get_norm_factors(data, steps: int, tqdm_kwargs: dict = {}) -> t.Tensor:
    """Per Section 3.1, find a fixed scalar factor so activation vectors have unit mean squared norm.
    This is very helpful for hyperparameter transfer between different layers and models.
    Use more steps for more accurate results.
    https://arxiv.org/pdf/2408.05147

    If experiencing troubles with hyperparameter transfer between models, it may be worth instead normalizing to the square root of d_model.
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes"""

    total_mean_squared_norm = None
    activation_layer_indices = None
    count = 0

    for _step, (act_BLD, layer_indices) in tqdm(
        zip(range(steps), data, strict=False),
        total=steps,
        desc="Calculating norm factor",
        **tqdm_kwargs,
    ):
        count += 1

        if act_BLD.ndim == 2:
            act_BLD = act_BLD.unsqueeze(1)

        assert act_BLD.shape[1] == len(layer_indices), (
            "Number of layers must match the number of layers in the data"
            f"act_BLD.shape[1]={act_BLD.shape[1]} is not equal to num_layers={len(layer_indices)}"
        )

        # (B, L, D) -> (B, L) -> (L)
        mean_squared_norm = t.mean(t.sum(act_BLD**2, dim=-1), dim=0)

        if total_mean_squared_norm is None:
            total_mean_squared_norm = mean_squared_norm
            activation_layer_indices = layer_indices
            continue

        total_mean_squared_norm += mean_squared_norm

    assert total_mean_squared_norm is not None, "No activations found"

    average_mean_squared_norm = total_mean_squared_norm / count
    norm_factors = t.sqrt(average_mean_squared_norm)

    print(f"Average mean squared norm: {average_mean_squared_norm}")
    print(f"Norm factors: {norm_factors}")

    return norm_factors, activation_layer_indices


def trainSAE(
    data,
    trainer_configs: list[dict],
    steps: int,
    trainer_names: list[str] | None = None,
    use_wandb: bool = False,
    wandb_entity: str = "",
    wandb_project: str = "",
    save_steps: Optional[list[int]] = None,
    save_dir: Optional[str] = None,
    log_steps: Optional[int] = None,
    activations_split_by_head: bool = False,
    transcoder: bool = False,
    run_cfg: dict = {},
    normalize_activations: bool = False,
    verbose: bool = False,
    device: str = "cuda",
    autocast_dtype: t.dtype = t.float32,
    backup_steps: Optional[int] = None,
    tqdm_kwargs: dict = {},
):
    """
    Train SAEs using the given trainers

    If normalize_activations is True, the activations will be normalized to have unit mean squared norm.
    The autoencoders weights will be scaled before saving, so the activations don't need to be scaled during inference.
    This is very helpful for hyperparameter transfer between different layers and models.

    Setting autocast_dtype to t.bfloat16 provides a significant speedup with minimal change in performance.
    """

    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_context = (
        nullcontext()
        if device_type == "cpu"
        else t.autocast(device_type=device_type, dtype=autocast_dtype)
    )

    if trainer_names is None:
        trainer_names = [f"trainer_{i}" for i in range(len(trainer_configs))]

    trainers = OrderedDict()
    for name, config in zip(trainer_names, trainer_configs):
        if "wandb_name" in config:
            config["wandb_name"] = f"{config['wandb_name']}_{name}"
        trainer_class = config.pop("trainer")
        trainers[name] = trainer_class(**config)

    wandb_processes = []
    log_queues = {}

    if use_wandb:
        # Note: If encountering wandb and CUDA related errors, try setting start method to spawn in the if __name__ == "__main__" block
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.set_start_method
        # Everything should work fine with the default fork method but it may not be as robust
        for name, trainer in trainers.items():
            log_queue = mp.Queue()
            log_queues[name] = log_queue
            wandb_config = trainer.config | run_cfg
            # Make sure wandb config doesn't contain any CUDA tensors
            wandb_config = {
                k: v.cpu().item() if isinstance(v, t.Tensor) else v
                for k, v in wandb_config.items()
            }
            wandb_process = mp.Process(
                target=new_wandb_process,
                args=(wandb_config, log_queue, wandb_entity, wandb_project),
            )
            wandb_process.start()
            wandb_processes.append(wandb_process)

    # make save dirs, export config
    if save_dir is not None:
        for name, trainer in trainers.items():
            trainer_save_dir = os.path.join(save_dir, name)
            os.makedirs(trainer_save_dir, exist_ok=True)
            # save config
            config = {"trainer": trainer.config}
            try:
                config["buffer"] = data.config
            except Exception:
                pass
            with open(os.path.join(trainer_save_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4)

    if normalize_activations:
        norm_factors, activation_layer_indices = get_norm_factors(
            data,
            steps=100,
            tqdm_kwargs=tqdm_kwargs,
        ).unsqueeze(-1)

        for name, trainer in trainers.items():
            relative_idx = activation_layer_indices.index(trainer.config["layer"])

            norm_factor = norm_factors[relative_idx].item()

            trainer.config["norm_factor"] = norm_factor
            # Verify that all autoencoders have a scale_biases method
            trainer.ae.scale_biases(1.0)
    else:
        norm_factors = t.ones((1,), device=device)

    for step, (act, layer_indices) in enumerate(tqdm(data, total=steps, **tqdm_kwargs)):
        act = act.to(dtype=autocast_dtype)

        if normalize_activations:
            act /= norm_factors

        if step >= steps:
            break

        # logging
        if (use_wandb or verbose) and step % log_steps == 0:
            log_stats(
                trainers,
                step,
                act,
                activations_split_by_head,
                transcoder,
                layer_indices,
                log_queues=log_queues,
                verbose=verbose,
            )

        # saving
        if save_steps is not None and step in save_steps and save_dir is not None:
            for name, trainer in trainers.items():
                trainer_save_dir = os.path.join(save_dir, name)
                relative_idx = layer_indices.index(trainer.config["layer"])
                norm_factor = norm_factors[relative_idx].item()

                if normalize_activations:
                    # Temporarily scale up biases for checkpoint saving
                    trainer.ae.scale_biases(norm_factor)

                if not os.path.exists(os.path.join(trainer_save_dir, "checkpoints")):
                    os.mkdir(os.path.join(trainer_save_dir, "checkpoints"))

                checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                t.save(
                    checkpoint,
                    os.path.join(trainer_save_dir, "checkpoints", f"ae_{step}.pt"),
                )

                if normalize_activations:
                    trainer.ae.scale_biases(1 / norm_factor)

        # backup
        if backup_steps is not None and step % backup_steps == 0:
            for name, trainer in trainers.items():
                trainer_save_dir = os.path.join(save_dir, name)
                # save the current state of the trainer for resume if training is interrupted
                # this will be overwritten by the next checkpoint and at the end of training
                t.save(
                    {
                        "step": step,
                        "ae": trainer.ae.state_dict(),
                        "optimizer": trainer.optimizer.state_dict(),
                        "config": trainer.config,
                        "norm_factor": norm_factor,
                    },
                    os.path.join(trainer_save_dir, "ae.pt"),
                )

        # training
        for name, trainer in trainers.items():
            with autocast_context:
                relative_idx = layer_indices.index(trainer.config["layer"])
                act = act[:, relative_idx]
                trainer.update(step, act)

    # save final SAEs
    for name, trainer in trainers.items():
        if normalize_activations:
            trainer.ae.scale_biases(norm_factor)

        if save_dir is not None:
            trainer_save_dir = os.path.join(save_dir, name)
            final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
            t.save(final, os.path.join(trainer_save_dir, "ae.pt"))

    # Signal wandb processes to finish
    if use_wandb:
        for queue in log_queues:
            queue.put("DONE")
        for process in wandb_processes:
            process.join()
