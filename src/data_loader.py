import json
import logging
import os

from typing import Any

import cv2

from .models import CameraData

CAMERA_TABLE = {
    "back_camera": "B_MIDRANGECAM_C",  # back camera
    "front_camera": "F_MIDLONGRANGECAM_CL",  # front camera CL
    "left_camera": "M_FISHEYE_L",  # left camera
    "right_camera": "M_FISHEYE_R"  # right camera
}


class ImgData:
    """
    Adatbetöltő a kalibrációs, kamera, és LIDAR adatoknak
    """

    def __init__(
        self,
        folder: str,
        section_id: str,
        frame_id: str,
        logger: logging.Logger
    ):
        """Init ImgData loggerrel együtt.

        Args:
            folder: Adatkönyvtár.
            section_id: Adott section azonosítója.
            frame_id: Adott frame száma.
            logger: Logger, a folyamat nyomonkövetésére.
        """
        self.folder = folder
        self.section_id = section_id
        self.frame_id = frame_id
        self.logger = logger

    def load_cal_data(self) -> dict[str, Any]:
        """JSON file-ból betölti a kamera adatokat.

        Returns:
            Dict ami tartalmazza a kalibrációs adatokat
        """
        path = os.path.join(
            self.folder,
            self.section_id,
            "sensor",
            "calibration",
            "calibration.json"
        )
        self.logger.info(f"Loading calibration parameters from: '{path}'")

        if not os.path.exists(path):
            self.logger.error(f"Calibration file does not exist: '{path}'")
            raise FileNotFoundError(f"Calibration file not found: {path}")

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in calibration file: '{path}' - {e}")
            raise

        self.logger.info(f"Calibration parameters loaded successfully: '{path}'")
        return data

    def load_camera(self) -> CameraData:
        """Kamera képek betöltése.

        Returns:
            CameraData, tartalmazza a képeket és egy idx(section_id, frame_id).
        """
        path = os.path.join(
            self.folder,
            self.section_id,
            "sensor",
            "camera"
        )

        if not os.path.exists(path):
            self.logger.error(f"Camera directory does not exist: '{path}'")
            raise FileNotFoundError(f"Camera directory not found: {path}")

        self.logger.info(f"Loading camera data from: '{path}'")

        cameras = {}
        for camera_name, camera_dir in CAMERA_TABLE.items():
            image_path = os.path.join(
                path,
                camera_dir,
                f"{camera_dir}{self.frame_id}.jpg"
            )
            self.logger.debug(f"Loading {camera_name} from: '{image_path}'")

            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to load {camera_name} image: '{image_path}'")
                raise ValueError(f"Could not load image: {image_path}")

            cameras[camera_name] = image
            self.logger.debug(f"{camera_name} loaded successfully")

        self.logger.info("All camera images loaded successfully")

        return CameraData(
            idx=(self.section_id, self.frame_id),
            back_camera=cameras["back_camera"],
            front_camera=cameras["front_camera"],
            left_camera=cameras["left_camera"],
            right_camera=cameras["right_camera"]
        )
