import cv2
import numpy as np
import pandas as pd

from src.logger import setup_logger
from src.data_loader import ImgData


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

    extrinsic = np.array(camera_data.params.extrinsic)  # body->cam (4x4)

    intrinsic = np.array(camera_data.params.intrinsic)  # projection matrix (3x4)

    # Homogenous LIDAR points (N, 4)
    lp_hom = np.hstack(
        [lidar_points, np.ones((lidar_points.shape[0], 1))]
    )

    points_cam = (extrinsic @ lp_hom.T)  # camera frame (4, N)

    print(points_cam.shape)

    valid = points_cam[2, :] > 0
    points_cam = points_cam[:, valid]
    depths = points_cam[2, :]

    uvw = intrinsic @ points_cam  # (3, N)
    uv = (uvw[:2] / uvw[2]).T  # pixel coords (N, 2)

    # Normalize depth for coloring
    d_min, d_max = depths.min(), depths.max()
    depths_norm = (depths - d_min) / (d_max - d_min + 1e-8)

    img = camera_data.data
    h, w = img.shape[:2]

    print(camera_data.params.model)

    # Draw on image
    for i, (u, v) in enumerate(uv):
        u, v = int(u), int(v)
        if 0 <= u < w and 0 <= v < h:
            d = depths_norm[i]
            color = (0, 255, 0) if d < 0.5 else (0, int(255 * (1 - d)), int(255 * d))
            cv2.circle(img, (u, v), 2, color, -1)

    cv2.imshow("LiDAR Projection", img)
    cv2.imwrite("LIDAR_projection.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
