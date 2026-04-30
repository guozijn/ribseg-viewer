import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "model" / "configs" / "multi_unetpp_b0_dice.yaml"


class Settings(BaseSettings):
    ribcxr_config: str = Field(
        default=os.path.normpath(str(_DEFAULT_CONFIG)),
        validation_alias="RIBCXR_CONFIG",
    )
    ribcxr_checkpoint: str = Field(default="", validation_alias="RIBCXR_CHECKPOINT")
    ribcxr_device: str = Field(default="cuda", validation_alias="RIBCXR_DEVICE")
    ribcxr_threshold: float = Field(default=0.5, validation_alias="RIBCXR_THRESHOLD")
    max_upload_mb: int = Field(default=32, validation_alias="RIBCXR_MAX_UPLOAD_MB")
    results_dir: str = Field(default="/tmp/ribseg_results", validation_alias="RIBCXR_RESULTS_DIR")

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
