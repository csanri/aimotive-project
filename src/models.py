from dataclasses import dataclass

import numpy as np


@dataclass
class CameraData:
    """Kép adatokhoz class.

    Attributes:
        idx: index a kép adatok betöltésére Tuple(section_id, frame_id).
        back_camera: Back camera képe, numpy array.
        front_camera: Front camera képe, numpy array.
        left_camera: Left camera képe, numpy array.
        right_camera: Right camera képe, numpy array.
    """
    idx: tuple[str, str]
    back_camera: np.ndarray
    front_camera: np.ndarray
    left_camera: np.ndarray
    right_camera: np.ndarray
