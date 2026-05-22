import os
import torch
import logging

from torch.utils.data import DataLoader
from torch.amp import GradScaler
from src.logger import setup_logger
from src.data_loader import Dataset
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from train import train_epoch

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

BASE_FOLDER = "/mnt/oldssd/aimotive-dataset/train/highway/"
CSV_PATH = "./data/id_data.csv"
OUTPUT_DIR = "batch_test"

EPOCHS = 10
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LR = 1e-5

if __name__ == "__main__":
    logger = setup_logger(level=logging.INFO)
    logger.info("Start")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device)

    logger.info("Model loaded")

    vram_used = torch.cuda.memory_allocated(device) / 1024**3
    logger.info(f"VRAM after loading the model: {vram_used:.2f} GB")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler("cuda")

    dataset = Dataset(csv_path=CSV_PATH, folder=BASE_FOLDER, logger=logger)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"==================== Epoch: {epoch}/{EPOCHS} ====================")
        train_loss = train_epoch(
            model=model,
            processor=processor,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            logger=logger
        )
        logger.info(f"Train loss: {train_loss:.4f} m")

        scheduler.step()
        logger.info(f"LR: {scheduler.get_last_lr()[0]:.2e}")

    # csak az első batch lekérése
    # batch_data = next(iter(train_loader))

    # images = batch_data['image']   # [5, 3, 704, 1024]
    # depths = batch_data['depth']   # [5, 3, 704, 1024]
    # masks = batch_data['gt_mask']  # [5, 3, 704, 1024]

    # logger.info("random frame:")

    # random_predictions = torch.rand_like(depths)*50.0

    # # mask l1 loss
    # loss = lossfunc(random_predictions, depths, masks)

    # images = images.to(device)
    # depths = depths.to(device)
    # masks = masks.to(device)

    # for i in range(5):
    #     img = images[i]
    #     depth = depths[i]
    #     mask = masks[i]

    #     logger.debug(f"Image data: {img}")
    #     logger.debug(f"Image shape: {img.shape}")

    #     logger.debug(f"Depth data: {depth}")
    #     logger.debug(f"Depth shape: {depth.shape}")

    #     logger.debug(f"Mask data: {mask}")
    #     logger.debug(f"Mask shape: {mask.shape}")

    #     prefix = f"sample_{i}"

    #     np.save(os.path.join(OUTPUT_DIR, f"{prefix}_depth_raw.npy"), depth)
    #     np.save(os.path.join(OUTPUT_DIR, f" {prefix}_mask_raw.npy"), mask)

    #     eredeti kép
    #     cv2.imwrite(os.path.join(OUTPUT_DIR, f"{prefix}_image.jpg"), img)

    #     mask
    #     mask_visual = (mask_np * 255).astype(np.uint8)
    #     cv2.imwrite(os.path.join(OUTPUT_DIR, f"{prefix}_mask.png"), mask_visual)
    # """
    #     #színes mélység
    #     depth_visual = np.zeros_like(img_np)
    #     if np.any(mask_np > 0):
    #         d_min, d_max = depth_np[mask_np > 0].min(), depth_np[mask_np > 0].max()
    #         depth_norm = 255 * (depth_np - d_min) / (d_max - d_min + 1e-8)
    #         depth_norm = depth_norm.astype(np.uint8)
    #         depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    #         depth_visual = cv2.bitwise_and(depth_color, depth_color, mask=mask_visual)

    #     cv2.imwrite(os.path.join(OUTPUT_DIR, f"{prefix}_depth_view.png"), depth_visual)
    # """
    # logger.info(f"Loss: {loss.item():.4f} méter")
    logger.info("Done")
