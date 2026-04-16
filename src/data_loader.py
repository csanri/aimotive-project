import logging

from .camera_loader import CameraDataLoader
from .lidar_loader import LidarDataLoader


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

    def __getitem__(self, key: str):
        """Access data using dictionary-like syntax.

        Args:
            key: Data key ('calibration', 'lidar', or camera name like 'front_camera')

        Returns:
            Requested data (lazy-loaded on first access)
        """
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
            raise KeyError(f"Unknown key: {key}. Valid keys: {["camera", 'calibration', 'lidar']}")


