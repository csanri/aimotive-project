import logging


def setup_logger(
    name: str = __name__,
    level: int = logging.DEBUG
) -> logging.Logger:
    """
    Ez a függvény létrehoz egy loggert,
    hogy nyomon tudjuk követni mi történik a folyamat során
    name: a pontos név ahol a logolás történik, alap változó a file neve
    level: logolt esemény szintje (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        datefmt="%Y/%m/%d - %H:%M:%S"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
