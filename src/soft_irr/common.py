from pathlib import Path


def silence_hf_logging() -> None:
    import os

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from huggingface_hub.utils import disable_progress_bars
    from transformers.utils import logging as hf_logging

    disable_progress_bars()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()


def load_dotenv(path="./.env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    with env_path.open() as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                key, value = stripped.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    import os

                    os.environ[key] = value
