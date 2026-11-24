import sys
import os

import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


class _Tee:
    """Mirrors stdout writes to both the terminal and a log file."""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", buffering=1, encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

    # Proxy all other attribute access to the real stdout so that libraries
    # that inspect sys.stdout (e.g. tqdm, rich) keep working.
    def __getattr__(self, name):
        return getattr(self.terminal, name)


@hydra.main(config_path="configs/", config_name="test.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.testing_pipeline import test

    # Redirect stdout → stdout.log inside the Hydra run directory.
    # os.getcwd() is already set to the Hydra output dir at this point.
    log_path = os.path.join(os.getcwd(), "stdout.log")
    tee = _Tee(log_path)
    sys.stdout = tee

    try:
        # Applies optional utilities
        utils.extras(config)

        # Evaluate model
        return test(config)
    finally:
        sys.stdout = tee.terminal
        tee.close()


if __name__ == "__main__":
    main()
