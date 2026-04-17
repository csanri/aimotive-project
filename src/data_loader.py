import logging

from .camera_loader import CameraDataLoader
from .lidar_loader import LidarDataLoader

VALID_KEYS = ["camera", "lidar"]


class ImgData:
    """
    Class Kamera és LIDAR adatok betöltésére
    """

    def __init__(
        self,
        folder: str,
        section_id: str,
        frame_id: str,
        logger: logging.Logger
    ):
        """Létrehozza az ImgData classt

        Parameters:
            folder: Az adatokat tartalmazó mappa elérési útja
            section_id: A section azonosítója
            frame_id: A frame azonosítója
            logger: Python logger a nyomonkövetésre
        """

        self.folder = folder
        self.section_id = section_id
        self.frame_id = frame_id
        self.logger = logger

    def __getitem__(self, key: str) -> CameraDataLoader | LidarDataLoader:
        """
        Dictionary a kamera és LIDAR adatokhoz

        Parameters:
            key: Az adattípus kulcsa ('camera', 'lidar')

        Returns:
            CameraDataLoader: ha key == 'camera'
            LidarDataLoader:  ha key == 'lidar'
        """

        self.logger.info(f"Fetching data: '{key}'")

        if key == 'lidar':
            return LidarDataLoader(
                folder=self.folder,
                section_id=self.section_id,
                frame_id=self.frame_id,
                logger=self.logger
            )
        elif key == "camera":
            return CameraDataLoader(
                folder=self.folder,
                section_id=self.section_id,
                frame_id=self.frame_id,
                logger=self.logger
            )
        else:
            self.logger.error(f"Unknown key: {key}. Valid keys: {VALID_KEYS}")
            raise KeyError(f"Unknown key: {key}. Valid keys: {VALID_KEYS}")
