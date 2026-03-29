import logging
import sys

import uvicorn

from app.config import load_settings
from app.utils import prepare_runtime


def main() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        prepare_runtime(settings)

        uvicorn.run(
            "app.http:app",
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level,
            loop="uvloop",
        )
    except RuntimeError:
        logger.error("Verification failed. Exiting.")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Manual interrupt received. Shutting down the server.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
