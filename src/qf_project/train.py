from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from torch import nn


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mse_loss = nn.MSELoss()(outputs["return"], batch["return_target"])
    bce_loss = nn.BCEWithLogitsLoss()(outputs["direction"], batch["direction_target"])
    ce_loss = nn.CrossEntropyLoss()(outputs["risk"], batch["risk_target"])
    total_loss = (
        loss_weights["regression"] * mse_loss
        + loss_weights["direction"] * bce_loss
        + loss_weights["risk"] * ce_loss
    )
    return total_loss, {
        "mse": mse_loss.detach(),
        "bce": bce_loss.detach(),
        "ce": ce_loss.detach(),
    }


def collect_predictions(model: torch.nn.Module, data_loader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    predictions = {"return": [], "direction": [], "risk": []}
    targets = {"return": [], "direction": [], "risk": [], "realized_return": []}

    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            outputs = model(features)
            predictions["return"].append(outputs["return"].cpu().numpy())
            predictions["direction"].append(torch.sigmoid(outputs["direction"]).cpu().numpy())
            predictions["risk"].append(torch.softmax(outputs["risk"], dim=1).cpu().numpy())
            targets["return"].append(batch["return_target"].cpu().numpy())
            targets["direction"].append(batch["direction_target"].cpu().numpy())
            targets["risk"].append(batch["risk_target"].cpu().numpy())
            targets["realized_return"].append(batch["realized_return"].cpu().numpy())

    return {
        "predictions": {key: np.concatenate(value) for key, value in predictions.items()},
        "targets": {key: np.concatenate(value) for key, value in targets.items()},
    }


def compute_validation_score(collected: dict[str, np.ndarray]) -> float:
    predictions = collected["predictions"]
    targets = collected["targets"]
    rmse = np.sqrt(mean_squared_error(targets["return"], predictions["return"]))
    direction_pred = (predictions["direction"] >= 0.5).astype(int)
    risk_pred = np.argmax(predictions["risk"], axis=1)
    direction_f1 = f1_score(targets["direction"], direction_pred, zero_division=0)
    risk_f1 = f1_score(targets["risk"], risk_pred, average="macro", zero_division=0)
    return direction_f1 + risk_f1 - rmse


def train_model(
    model: torch.nn.Module,
    dataloaders,
    device: torch.device,
    training_config: dict,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    best_state = deepcopy(model.state_dict())
    best_score = -float("inf")
    patience = 0
    history: list[dict[str, float]] = []

    model.to(device)

    for epoch in range(training_config["epochs"]):
        model.train()
        epoch_losses = []
        for batch in dataloaders["train"]:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch["features"])
            total_loss, _ = compute_losses(outputs, batch, training_config["loss_weights"])
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config["gradient_clip_norm"])
            optimizer.step()
            epoch_losses.append(total_loss.item())

        validation_predictions = collect_predictions(model, dataloaders["val"], device)
        validation_score = compute_validation_score(validation_predictions)
        average_train_loss = float(np.mean(epoch_losses))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": average_train_loss,
                "validation_score": validation_score,
            }
        )

        if validation_score > best_score:
            best_score = validation_score
            best_state = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= training_config["early_stopping_patience"]:
                break

    model.load_state_dict(best_state)
    return model, history


def summarize_predictions(collected: dict[str, np.ndarray]) -> dict[str, float]:
    predictions = collected["predictions"]
    targets = collected["targets"]
    direction_pred = (predictions["direction"] >= 0.5).astype(int)
    risk_pred = np.argmax(predictions["risk"], axis=1)
    return {
        "rmse": float(np.sqrt(mean_squared_error(targets["return"], predictions["return"]))),
        "mae": float(mean_absolute_error(targets["return"], predictions["return"])),
        "direction_accuracy": float(accuracy_score(targets["direction"], direction_pred)),
        "direction_f1": float(f1_score(targets["direction"], direction_pred, zero_division=0)),
        "risk_macro_f1": float(f1_score(targets["risk"], risk_pred, average="macro", zero_division=0)),
    }
