import json
import logging
import os
import numpy as np
import cv2

from typing import Any

CAMERA_TABLE = {
    "back_camera":  "B_MIDRANGECAM_C",       # back camera
    "front_camera": "F_MIDLONGRANGECAM_CL",  # front camera CL
    "left_camera":  "M_FISHEYE_L",           # left camera
    "right_camera": "M_FISHEYE_R"            # right camera
}


class ImgData:
    def __init__(
        self,
        folder: str,
        section_id: str,
        frame_id: str,
        logger: logging.Logger
    ):
        """
        Args:
            folder: Fő útvonal elérése
            section_id: sectio_id [highway, night, rain, urban]
            frame_id: addott mérési tartomáyn mappájának neve
            logger: betöltött logger
        """
        self.folder = folder
        self.section_id = section_id
        self.frame_id = frame_id
        self.logger = logger

    def load_data(self) -> dict[str, Any]:
        """
        Betölti a kalibrációs adatokat
        Return:
            dict: tartalmazza az összes kalibrációs adatot
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
            self.logger.error(f"JSON file does not exist: '{path}'")
            raise FileNotFoundError

        with open(path, 'r') as f:
            data = json.load(f)

        self.logger.info(f"Calibration parameters loaded: '{path}'")

        return data

    def load_camera(self) -> list[np.ndarray]:
        """
        Betölti a kamera képeket
        Return:
            list[np.ndarray]: 4 kamera kép adott frame_id-hoz
        """
        path = os.path.join(
            self.folder,
            self.section_id,
            "sensor/camera"
        )

        if not os.path.exists(path):
            self.logger.error(f"Camera path does not exist: '{path}'")
            raise FileNotFoundError

        self.logger.info(f"Loading camera data from: '{path}'")

        self.back_camera = cv2.imread(
            os.path.join(
                path,
                CAMERA_TABLE["back_camera"],
                CAMERA_TABLE["back_camera"] + self.frame_id + ".jpg"
            )
        )

        self.front_camera = cv2.imread(
            os.path.join(
                path,
                CAMERA_TABLE["front_camera"],
                CAMERA_TABLE["front_camera"] + self.frame_id + ".jpg"
            )
        )

        self.left_camera = cv2.imread(
            os.path.join(
                path,
                CAMERA_TABLE["left_camera"],
                CAMERA_TABLE["left_camera"] + self.frame_id + ".jpg"
            )
        )

        self.right_camera = cv2.imread(
            os.path.join(
                path,
                CAMERA_TABLE["right_camera"],
                CAMERA_TABLE["right_camera"] + self.frame_id + ".jpg"
            )
        )

        return [self.back_camera, self.front_camera, self.left_camera, self.right_camera]
