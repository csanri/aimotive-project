import json
import logging
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from torchvision import transforms
import os


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
        path = Path(self.folder
                    + '/'
                    + self.section_id
                    + '/'
                    + self.frame_id
                    + "/sensor/calibration/calibration.json"
                    )
        self.logger.info(f"Loading calibration parameters from: '{path}'")

        if not path.exists():
            self.logger.error(f"JSON file does not exist: '{path}'")
            raise FileNotFoundError

        with path.open() as f:
            data = json.load(f)

        self.logger.info(f"Calibration parameters loaded: '{path}'")

        return data

    def load_camera(self, camera: str, img_name: str) -> np.ndarray:
        """
        Bertölti az adott kamera adott képét
        Return:
            np.ndarray: kamera képe Numpy arrayként
            (Height, Width, Channel)
        """
        path = Path(self.folder
                    + '/'
                    + self.section_id
                    + '/'
                    + self.frame_id
                    + "/sensor/camera/"
                    + camera
                    + '/'
                    )
        self.logger.info(f"Loading camera data from: '{path}'")

        if not path.exists():
            self.logger.error(f"Camera path does not exist: '{path}'")
            raise FileNotFoundError

        img = Image.open(os.path.join(path, img_name))

        # Transzformálni kell a képet torch tenzorból numpy arraybe
        transform = transforms.ToTensor()
        tensor = transform(img)
        array = tensor.numpy().transpose(1, 2, 0)  # Height, Width, Channel

        return (array * 255).astype(np.uint8)
