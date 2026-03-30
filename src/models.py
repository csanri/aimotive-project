from dataclasses import dataclass
from typing import Any
import nump as np


@dataclass
class ImgData:
    calibration: dict[str, Any]
    front_cam:   np.ndarray
    lidar:       np.ndarray
