import pandas as pd
import numpy as np
import os
import torch
import random

from .camera_loader import CameraDataLoader
from .lidar_loader import LidarDataLoader
from .projection import CameraProjection


class Dataset:
    def __init__(self, csv_path: str, folder: str, logger):
        self.folder = folder
        self.logger = logger

        df = pd.read_csv(csv_path, dtype=str, index_col=0)

        self.logger.debug(f"Checking: {os.path.join(self.folder, df['section_id'].iloc[0])}")
        self.logger.debug(f"Exists: {os.path.exists(os.path.join(self.folder, df['section_id'].iloc[0]))}")

        # csak azokat a sorokat tartjuk meg, ahol a mappa létezik
        self.valid_ids = []
        self.logger.debug("Filtering csv for existing files...")

        _folder = folder
        mask = df["section_id"].apply(
            lambda s: os.path.exists(os.path.join(_folder, s))
        )
        self.logger.debug(f"Mask true count: {mask.sum()}")

        self.valid_df = df[mask].reset_index(drop=True)
        self.logger.debug(f"Filtering: {len(self.valid_df)} valid samples.")

    def __len__(self):
        return len(self.valid_df)

    def __getitem__(self, idx: int):
        # ha egy minta betöltése sikertelen (pl. hiányzó fájl),
        # véletlenszerű másik mintát próbálunk helyette
        for attempt in range(10):
            try:
                return self._load_sample(idx)
            except FileNotFoundError as e:
                self.logger.warning(
                    f"Hiányzó fájl (idx={idx}, attempt={attempt+1}): {e}"
                )
                idx = random.randrange(len(self.valid_df))
            except Exception as e:
                self.logger.warning(
                    f"Betöltési hiba (idx={idx}, attempt={attempt+1}): {e}"
                )
                idx = random.randrange(len(self.valid_df))

        raise RuntimeError("Failed loading sample")

    def _load_sample(self, idx: int) -> dict:
        row = self.valid_df.iloc[idx]
        section_id = row["section_id"]
        frame_id = row["frame_id"]

        # szenzorok betöltése
        lidar_loader = LidarDataLoader(
            self.folder,
            section_id,
            frame_id,
            self.logger
        )
        camera_loader = CameraDataLoader(
            self.folder,
            section_id,
            frame_id,
            self.logger
        )

        cam = camera_loader.front_camera
        img = cam.data  # [H,W,3]
        h, w = img.shape[:2]

        projection = CameraProjection(
            image=img,
            extrinsic=cam.params.extrinsic,
            intrinsic=cam.params.intrinsic,
            lidar_points=lidar_loader.data.data,
            logger=self.logger
        )

        img = img.astype(np.uint8)
        img = img.transpose(2, 0, 1)  # [3, H, W]
        img = torch.from_numpy(img)
        img = img.float() / 255.

        proj_matrix = projection.get_projection_matrix()
        depth_gt, depth_mask = self.rasterize_to_image(proj_matrix, h, w)

        depth_gt = depth_gt.transpose(2, 0, 1)
        depth_gt = torch.from_numpy(depth_gt)

        depth_mask = (depth_mask > 0).astype(np.float32)
        depth_mask = depth_mask.transpose(2, 0, 1)
        depth_mask = torch.from_numpy(depth_mask)

        # (H, W, C)
        return {
            'image': img,
            'depth': depth_gt,
            'gt_mask': depth_mask
        }

    def rasterize_to_image(self, proj_matrix, h, w):
        """
        A 2D-re vetített pontokból sűrűbb (rasterized) mátrixot készít.
        """
        depth_map = np.zeros((h, w, 1), dtype=np.float32)
        mask = np.zeros((h, w, 1), dtype=np.float32)

        if proj_matrix is not None and len(proj_matrix) > 0:
            # koordináták kerekítése egészekre a pixelekhez
            u = np.round(proj_matrix[:, 0]).astype(int)
            v = np.round(proj_matrix[:, 1]).astype(int)
            depths = proj_matrix[:, 2]

            # csak a képkereten belüli pontokat tartjuk meg
            valid_indices = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u, v, depths = u[valid_indices], v[valid_indices], depths[valid_indices]

            # értékek beírása a mátrixokba
            depth_map[v, u, 0] = depths
            mask[v, u, 0] = 1.0

        return depth_map, mask
