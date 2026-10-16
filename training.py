import os
import tempfile
import random

import numpy as np
import torch
import shutup
import sys

import torch.optim as optim
import statistics as s

import config.config as cfg

from utils.stopper import PatienceStopper

from provider.dataset_provider import get_loader
from model.dense121_unet.dense121_unet_model import DenseNet121
from model.unet.unet_model import UNET

from loss.dice_sb_loss import DiceSBLoss

from metrics.mean_errors import mae, r_mse, medae
from metrics.similarity import ssim, zncc

from ray import tune
from ray import train as raytrain
from ray.train import Checkpoint
from ray.experimental.tqdm_ray import tqdm

sys.path.append(os.getcwd())
device = cfg.get_device()
shutup.please()


def train(loader, loss_fn, optimizer, scaler, model, aux):
    torch.enable_grad()
    model.train()

    running_loss = []
    running_mae = []
    running_medae = []
    running_mse = []
    running_rmse = []
    running_zncc = []
    running_ssim = []

    running_total = []

    loop = tqdm(loader)

    for (data, target, degree_target, distance_target, _) in loop:
        optimizer.zero_grad(set_to_none=True)
        data = data.to(device)

        data, degree, distance = model(data)
        data = torch.sigmoid(data)

        target = target.to(device)
        if aux:
            degree_target = degree_target.to(device)
            distance_target = distance_target.to(device)

        with torch.cuda.amp.autocast():
            loss = loss_fn(data, target)
            loss_value = loss.item()

            if aux:
                loss += loss_fn(degree, degree_target) + loss_fn(distance_target, distance_target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        mse, rmse = r_mse(data, target)

        running_mae.append(mae(data, target).item())
        running_medae.append(medae(data, target).item())
        running_mse.append(mse.item())
        running_rmse.append(rmse.item())
        running_zncc.append(zncc(data, target).item())
        running_ssim.append(ssim(data, target).item())

        running_loss.append(loss_value)

        if aux:
            loss_total = loss.item()
            running_total.append(loss_total)
        else:
            running_total.append(0)

    return (s.mean(running_loss), s.mean(running_mae), s.mean(running_medae), s.mean(running_mse), s.mean(running_rmse),
            s.mean(running_zncc), s.mean(running_ssim), s.mean(running_total))


def valid(loader, loss_fn, model, aux):
    model.eval()

    running_loss = []
    running_mae = []
    running_medae = []
    running_mse = []
    running_rmse = []
    running_zncc = []
    running_ssim = []

    running_total = []

    loop = tqdm(loader)

    for (data, target, degree_target, distance_target, _) in loop:
        data = data.to(device)

        data, degree, distance = model(data)
        data = torch.sigmoid(data)

        target = target.to(device)
        if aux:
            degree_target = degree_target.to(device)
            distance_target = distance_target.to(device)

        with torch.no_grad():
            loss = loss_fn(data, target)
            loss_value = loss.item()

            if aux:
                loss += loss_fn(degree, degree_target) + loss_fn(distance_target, distance_target)

        mse, rmse = r_mse(data, target)

        running_mae.append(mae(data, target).item())
        running_medae.append(medae(data, target).item())
        running_mse.append(mse.item())
        running_rmse.append(rmse.item())
        running_zncc.append(zncc(data, target).item())
        running_ssim.append(ssim(data, target).item())

        running_loss.append(loss_value)

        if aux:
            loss_total = loss.item()
            running_total.append(loss_total)
        else:
            running_total.append(0)

    return (s.mean(running_loss), s.mean(running_mae), s.mean(running_medae), s.mean(running_mse), s.mean(running_rmse),
            s.mean(running_zncc), s.mean(running_ssim), s.mean(running_total))


def run(ray_config):
    torch.cuda.empty_cache()

    set_reproducibility()

    aux = ray_config["aux"]

    model = DenseNet121(in_channels=3, out_channels=1,
                        attention=ray_config["attention"], aspp=ray_config["aspp"], aux=ray_config["aux"])

    optimizer = optim.Adam(model.parameters(), lr=ray_config["lr"])
    loss_fn = DiceSBLoss()
    scaler = torch.cuda.amp.GradScaler()

    epochs_done = 0

    model.to(device)

    train_loader = get_loader(cfg.get_train_path(version=ray_config["frames"]),
                              ray_config["batch_size"], cfg.num_workers(), cfg.pin_memory())

    valid_loader = get_loader(cfg.get_validation_path(version=ray_config["frames"]),
                              ray_config["batch_size"], cfg.num_workers(), cfg.pin_memory())

    for epoch in range(epochs_done + 1, 101):
        t_loss, t_mae, t_medae, t_mse, t_rmse, t_zncc, t_ssim, t_total = train(
            train_loader, loss_fn, optimizer, scaler, model, aux)

        v_loss, v_mae, v_medae, v_mse, v_rmse, v_zncc, v_ssim, v_total = valid(
            valid_loader, loss_fn, model, aux)

        # Ray report
        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        metrics = {
            "training_loss": t_loss,
            "validation_loss": v_loss,
            "training_mae": t_mae,
            "training_medae": t_medae,
            "training_mse": t_mse,
            "training_rmse": t_rmse,
            "training_zncc": t_zncc,
            "training_ssim": t_ssim,
            "validation_mae": v_mae,
            "validation_medae": v_medae,
            "validation_mse": v_mse,
            "validation_rmse": v_rmse,
            "validation_zncc": v_zncc,
            "validation_ssim": v_ssim,
            "training_total": t_total,
            "validation_total": v_total
        }

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                checkpoint_data,
                os.path.join(temp_checkpoint_dir, "model_epoch{}.pt".format(epoch)),
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            raytrain.report(metrics, checkpoint=checkpoint)


def set_reproducibility():
    torch.manual_seed(251199)
    np.random.seed(251199)
    random.seed(251199)
    torch.use_deterministic_algorithms(True)



if __name__ == '__main__':
    # scheduler = ASHAScheduler(max_t=100, grace_period=1, reduction_factor=2)

    stopper = PatienceStopper(
        metric="validation_loss",
        mode="min",
        patience=5
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(run),
            resources={"cpu": 16, "gpu": 1},
        ),
        tune_config=tune.TuneConfig(
            metric="validation_loss",
            mode="min",
            num_samples=1,
        ),
        param_space=cfg.get_ray_config(),
        run_config=raytrain.RunConfig(
            stop=stopper,
            local_dir=os.path.join(cfg.get_raw_data_path(), "ray_results"),
        ),
    )
    results = tuner.fit()

    best_result = results.get_best_result("validation_loss", "min", "all")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["validation_loss"]))
