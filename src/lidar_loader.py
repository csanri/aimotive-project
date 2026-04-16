import os
import laspy
import logging
import json
import numpy as np

from .models import LidarData


class LidarDataLoader:
    def __init__(
        self,
        folder: str,
        section_id: str,
        frame_id: str,
        logger: logging.Logger
    ):
        self.folder = folder
        self.section_id = section_id
        self.frame_id = frame_id
        self.logger = logger
        self.data = self.load_lidar()

    def _load_lidar_internal(self) -> LidarData:
        """Internal method to load LIDAR data."""
        path = self._get_lidar_path()

        try:
            las = laspy.read(path)
            lidar_pcd = np.array(
                [las.x, las.y, las.z, las.intensity, las.gps_time],
                dtype=np.float32)
            lidar_pcd = lidar_pcd.T
        except Exception as e:
            self.logger.error(f"Error reading LIDAR data: {path} - {e}")
            raise

        if las is None:
            self.logger.error(f"LIDAR file is empty: {path}")
            raise

        egomotion_path = os.path.join(
            self.folder,
            self.section_id,
            'sensor',
            'gnssins',
            'egomotion.json'
        )

        with open(egomotion_path) as f:
            egomotion = json.load(f)

        RT_main_frame = np.array(
            egomotion[str(int(self.frame_id))]
        ).reshape(4, 4)

        RT_current = np.array(egomotion[str(int(self.frame_id))]).reshape(4, 4)
        RT_transform = np.linalg.inv(RT_main_frame) @ RT_current

        lidar_data = self.filter_ego_car(lidar_pcd)
        lidar_data_coords = np.hstack(
            [lidar_data[:, :3],
             np.ones((lidar_data.shape[0], 1))]
        )
        lidar_data[:, :3] = (lidar_data_coords @ RT_transform.T)[:, :3]

        return LidarData(
            idx=(self.section_id, self.frame_id),
            data=lidar_data
        )

    def filter_ego_car(self, pc):
        filter_x = np.logical_and(pc[:, 0] < 3.8, pc[:, 0] > -1.2)
        filter_y = np.logical_and(pc[:, 1] < 1.7, pc[:, 1] > -1.7)
        valid = ~np.logical_and(filter_x, filter_y)
        return pc[valid]

    def load_lidar(self):
        """Load LIDAR data."""
        return self._load_lidar_internal()

    def _get_lidar_path(self) -> str:
        """Get LIDAR file path."""
        return os.path.join(
            self.folder,
            self.section_id,
            "dynamic",
            "raw-revolutions",
            f"frame_{self.frame_id}.laz"
        )
