import os
import cv2
import logging
import json
import numpy as np

from typing import Any

from .models import CameraData, CameraParams

CAMERA_TABLE = {
    "back_camera": "B_MIDRANGECAM_C",        # back camera
    "front_camera": "F_MIDLONGRANGECAM_CL",  # front camera CL
    "left_camera": "M_FISHEYE_L",            # left camera
    "right_camera": "M_FISHEYE_R"            # right camera
}


class CameraDataLoader:
    """
    Class Kamera képek és kalibrációs paraméterek betöltéséért

    Automatikusan betölti mind a négy kamera képét
    (front, back, left, right), valamint a hozzájuk paramétereket

    Attributes:
        front_camera (CameraData): Első kamera adata és paraméterei
        back_camera (CameraData): Hátsó kamera adata és paraméterei
        left_camera (CameraData): Bal oldali kamera adata és paraméterei
        right_camera (CameraData): Jobb oldali kamera adata és paraméterei
    """

    def __init__(
        self,
        folder: str,
        section_id: str,
        frame_id: str,
        logger: logging.Logger
    ):
        """
        Létrehozza a CameraDataLoader classt és betölti az összes kamerát

        Parameters:
            folder: Az adatokat tartalmazó mappa elérési útja
            section_id: A section azonosítója
            frame_id: A frame azonosítója
            logger: Python logger nyomonkovetésre
        """
        self.folder = folder
        self.section_id = section_id
        self.frame_id = frame_id
        self.logger = logger

        self.front_camera = CameraData(
            idx=(section_id, frame_id),
            camera_id="front_camera",
            data=self.load_camera("front_camera"),
            params=self.read_camera_params("front_camera")
        )

        self.back_camera = CameraData(
            idx=(section_id, frame_id),
            camera_id="back_camera",
            data=self.load_camera("back_camera"),
            params=self.read_camera_params("back_camera")
        )

        self.left_camera = CameraData(
            idx=(section_id, frame_id),
            camera_id="left_camera",
            data=self.load_camera("left_camera"),
            params=self.read_camera_params("left_camera")
        )

        self.right_camera = CameraData(
            idx=(section_id, frame_id),
            camera_id="right_camera",
            data=self.load_camera("right_camera"),
            params=self.read_camera_params("right_camera")
        )

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def load_camera(self, camera_id: str) -> np.ndarray:
        """
        Betölti a megadott kamera képét NumPy tömbként

        Parameters:
            camera_id: A kamera azonosítója (pl. 'front_camera')

        Returns:
            np.ndarray: A betöltött kép (H, W, 3)
        """
        return self._load_camera_internal(camera_id)

    def read_camera_params(self, camera: str) -> CameraParams:
        """
        Beolvassa és visszaadja a megadott kamera kalibrációs paramétereit

        Parameters:
            camera_id: A kamera azonosítója (pl. 'front_camera').

        Returns:
            CameraParams: A kamera modell, intrinsic, extrinsic és
                          distortion paramétereket
        """

        params = self._load_calibration_data()[CAMERA_TABLE[camera]]
        model = params["model"]
        intrinsic = self._get_intrinsic(
            params["focal_length_px"],
            params["principal_point_px"])
        extrinsic = params["RT_sensor_from_body"]
        dist = np.array(
            params.get("distortion_coeffs", [0, 0, 0, 0]),
            dtype=np.float32
        )

        return CameraParams(
            model=model,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            dist=dist
        )

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------

    def _load_camera_internal(self, camera_id: str) -> np.ndarray:
        """
        Betölti a kamera képét

        Parameters:
            camera_id: A kamera azonosítója (pl. 'front_camera')

        Returns:
            np.ndarray: A betöltött kép
        """

        camera = CAMERA_TABLE[camera_id]
        path = self._get_camera_path(camera)

        if not os.path.exists(path):
            self.logger.error(f"Camera directory does not exist: '{path}'")
            raise FileNotFoundError(f"Camera directory not found: {path}")

        self.logger.info(f"Loading camera data from: '{path}'")

        try:
            image = cv2.imread(str(path))
        except Exception as e:
            self.logger.error(f"Failed to read image data: {path} - {e}")
            raise

        if image is None:
            self.logger.error(f"Failed to load {camera} image: '{path}'")
            raise ValueError(f"Could not load image: {path}")

        self.logger.debug(f"{camera} loaded successfully")

        return image

    def _get_camera_path(self, camera: str) -> str:
        """
        Összeállítja a kamera képfájl teljes elérési útját

        Parameters:
            camera: A kamera belső azonosítója

        Returns:
            str: A képfájl teljes elérési útja
        """

        return os.path.join(
            self.folder,
            self.section_id,
            "sensor",
            "camera",
            camera,
            f"{camera}_{self.frame_id}.jpg"
        )

    def _get_intrinsic(self, f: list[float], p: list[float]) -> np.ndarray:
        """
        Összeállítja a 3x4-es intrinsic projekciós mátrixot

        Parameters:
            f: Fókusztávolság pixelben [fx, fy]
            p: Főpont koordinátái pixelben [cx, cy]

        Returns:
            np.ndarray: 3x4-es intrinsic projekciós mátrix
        """

        proj_matrix = np.array([
            [f[0], 0, p[0], 0],
            [0, f[1], p[1], 0],
            [0, 0, 1, 0]
        ])

        return proj_matrix

    def _load_calibration_data(self) -> dict[str, Any]:
        """
        Betölti és visszaadja a kalibrációs JSON fájl tartalmát

        Returns:
            dict: A kalibrációs fájl teljes tartalma dictionaryként
        """

        path = self._get_calibration_path()
        self.logger.info(f"Loading radar parameters from: '{path}'")

        if not os.path.exists(path):
            self.logger.error(f"Radar file does not exist: '{path}'")
            raise FileNotFoundError(f"Radar file not found: {path}")

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in radar file: '{path}' - {e}")
            raise

        self.logger.info(f"Radar parameters loaded successfully: '{path}'")
        return data

    def _get_calibration_path(self) -> str:
        """
        Összeállítja a kalibrációs JSON fájl teljes elérési útját

        Returns:
            str: A kalibrációs fájl teljes elérési útja
        """

        return os.path.join(
            self.folder,
            self.section_id,
            "sensor",
            "calibration",
            "calibration.json"
        )
