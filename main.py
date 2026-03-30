from src.logger import setup_logger
from src.load_json import load_json
from src.models import ImgData

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Loading Project Files")

    data = ImgData()

    data.calibration = load_json("/mnt/oldssd/train/rain/20210427-101009-00.57.15-00.57.30@Jarvis/sensor/calibration/calibration.json", logger)

    for k, v in data.params.items():
        print(k, v)
