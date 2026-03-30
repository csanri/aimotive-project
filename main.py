from src.logger import setup_logger
from src.data_loader import ImgData

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Loading Project Files")

    # Példa adatok
    img_data = ImgData(
        folder="/mnt/oldssd/train/",
        section_id="highway",
        frame_id="20210401-073402-00.18.00-00.18.15@Jarvis",
        logger=logger
    )

    calibration_data = img_data.load_data()
    camera_data = img_data.load_camera(
        camera="B_MIDRANGECAM_C",
        img_name="B_MIDRANGECAM_C_0016239.jpg"
    )

    print(f"Calibration data: {calibration_data}")
    print(f"Camera data: {camera_data}")
