import numpy as np
import pandas as pd

from src.logger import setup_logger
from src.data_loader import ImgData
from src.projection import CameraProjection


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Loading Project Files")

    ids = pd.read_csv("./data/id_data.csv", dtype=str)

    section_id = ids.iloc[1, 1]
    frame_id = ids.iloc[1, 2]

    logger.debug(f"{section_id}, {frame_id}")

    img_data = ImgData(
        folder="/mnt/oldssd/train/highway/",
        section_id=section_id,
        frame_id=frame_id,
        logger=logger
    )

    camera_data = img_data["camera"].front_camera
    lidar_points = img_data["lidar"].data.data[:, :3]

    extrinsic = np.array(camera_data.params.extrinsic)  # body -> cam (4, 4)
    intrinsic = np.array(camera_data.params.intrinsic)  # projection matrix (3, 4)

    projection = CameraProjection(
        image=camera_data.data,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        lidar_points=lidar_points,
        logger=logger
    )

    projection.show_points_on_img()
