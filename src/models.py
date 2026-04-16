from dataclasses import dataclass

import numpy as np


@dataclass
class CameraParams:
    model: str
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    dist: np.array


@dataclass
class CameraData:
    """Kép adatokhoz class.

    Attributes:
        idx: index a kép adatok betöltésére Tuple(section_id, frame_id).
        camera_id: melyik kamera képéhez tartozó adatokat látjuk (front, back, left, right)
        data: kép adatok ndarray-ként
    """
    idx: tuple[str, str]
    camera_id: str
    data: np.ndarray
    params: CameraParams


@dataclass
class LidarData:
    idx: tuple[str, str]
    data: np.ndarray
