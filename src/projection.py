import cv2
import logging
import numpy as np


class CameraProjection:
    """
    class a LIDAR pontfelhő kamera képre vetítéséért.

    LIDAR pontok transzformációját a kamera koordináta-rendszerébe,
    majd a kamera belső paraméterei segítségével pixelkoordinátákra vetíti
    Az eredményt mélység szerinti színkódolással jeleníti meg a képen

    Attributes:
        image (np.ndarray): A kamera képe (H, W, 3)
        extrinsic (np.ndarray): test -> kamera transzformáció (4, 4)
        intrinsic (np.ndarray): Projekciós mátrix (3, 4)
        lidar_points (np.ndarray): LIDAR pontok (N, 5)
    """

    def __init__(
        self,
        image: np.array,
        extrinsic: np.array,
        intrinsic: np.array,
        lidar_points: np.array,
        logger: logging.Logger
    ):
        """
        Létrehozza a CameraProjection objektumot

        Parameters:
            image: A kamera képe NumPy tömbként (H, W, 3)
            extrinsic: 4x4-es extrinsic mátrix (test -> kamera)
            intrinsic: 3x4-es intrinsic projekciós mátrix
            lidar_points: LIDAR pontok (N, 5)
            logger: logger
        """
        self.image = image
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.lidar_points = lidar_points
        self.logger = logger

        self.logger.info(
            f"CameraProjection init"
            f"Image size: {image.shape}, "
            f"no. LIDAR points: {lidar_points.shape[0]}"
        )

    def lidar_to_camera(self) -> tuple[np.ndarray, np.ndarray]:
        """
        A LIDAR pontokat kamera pixelkoordinátákra vetíti

        Returns:
            uv (np.ndarray): Vetített pixelkoordináták (N, 2)
            depths (np.ndarray): Kamera Z-tengelyen mélysé (N, )
        """

        self.logger.info("LIDAR -> kamera transformation started")

        lp = self.lidar_points[:, :3]
        self.logger.debug(f"no. LIDAR points: {lp.shape[0]}")

        lp_hom = np.hstack([lp, np.ones((lp.shape[0], 1))])
        points_cam = self.extrinsic @ lp_hom.T  # (4, N)

        # Csak a kamera előtt lévő pontok (Z > 0)
        valid = points_cam[2, :] > 0
        points_cam_valid = points_cam[:, valid]

        uvw = self.intrinsic @ points_cam_valid  # (3, N)
        uv = (uvw[:2] / uvw[2]).T                # pixel coords (N, 2)
        depths = points_cam_valid[2, :]

        self.logger.debug(
            f"Projection done – {uv.shape[0]}"
            f"Depth range: [{depths.min():.2f}, {depths.max():.2f}]"
        )

        return (uv, depths)

    def show_points_on_img(self):
        """
        A vetített LIDAR pontokat mélység szerint színkódolva rajzolja a képre

        Color coding:
            - Közeli pontok (normalizált mélység < 0.5): zöld
            - Távoli pontok (normalizált mélység >= 0.5): zöldtől pirosba
        """
        self.logger.info("Drawing LIDAR points on image")
        uv, depths = self.lidar_to_camera()
        h, w = self.image.shape[:2]

        self.logger.debug(f"Image dimensions: {w}x{h} px")

        # Depth normlizálása színkódolásra
        d_min, d_max = depths.min(), depths.max()
        depths_norm = (depths - d_min) / (d_max - d_min + 1e-8)

        # LIDAR pontok rávetítése a képre
        for i, (u, v) in enumerate(uv):
            u, v = int(u), int(v)
            if 0 <= u < w and 0 <= v < h:
                d = depths_norm[i]
                color = (0, 255, 0) if d < 0.5 else (0, int(255 * (1 - d)), int(255 * d))
                cv2.circle(self.image, (u, v), 2, color, -1)

        output_path = "LIDAR_projection.jpg"

        cv2.imshow("LIDAR Projection", self.image)
        cv2.imwrite("LIDAR_projection.jpg", self.image)
        self.logger.info(f"Image saved: '{output_path}'")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.logger.debug("OpenCV window closed")
