import os
import torch
import logging
import segmentation_models_pytorch as smp
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
from src.logger import setup_logger
from src.data_loader import Dataset
from train import validate_epoch

VAL_BASE_FOLDER = "/mnt/oldssd/aimotive-dataset/val/"
CSV_PATH = "./data/id_data_val.csv"
DATA_DIRS = ["highway", "night", "rain", "urban"]
BATCH_SIZE = 7
CHECKPOINT_PATH = "weights/epoch_005_loss_4.5506.pt"
VAL_LOSSES_CSV = "val_losses.csv"

if __name__ == "__main__":
    logger = setup_logger(level=logging.INFO)
    logger.info("Validation start")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Checkpoint betöltve: {CHECKPOINT_PATH} (epoch {checkpoint['epoch']})")

    val_datasets = [
        Dataset(
            csv_path=CSV_PATH,
            folder=os.path.join(VAL_BASE_FOLDER, d),
            logger=logger
        ) for d in DATA_DIRS
    ]
    combined_val_dataset = ConcatDataset(val_datasets)
    val_loader = DataLoader(
        combined_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=12,
        pin_memory=True
    )

    logger.info(f"Validation set mérete: {len(combined_val_dataset)}")
    logger.info(f"Batch-ek száma: {len(val_loader)}")

    losses_per_batch, val_loss = validate_epoch(
        model=model,
        val_loader=val_loader,
        device=device,
        logger=logger
    )
    logger.info(f"Validation loss (átlag): {val_loss:.4f} m")

    pd.DataFrame({
        "batch":     range(1, len(losses_per_batch) + 1),
        "val_loss":  losses_per_batch
    }).to_csv(VAL_LOSSES_CSV, index=False)
    logger.info(f"Batch-szintű veszteségek mentve: {VAL_LOSSES_CSV}")
