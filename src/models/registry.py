"""Central Model Registry and Multi-Model Orchestration Layer.

Manages model lifecycle: training, artifact persistence, versioned metadata,
loading for API inference, and schema harmonization across LightGBM, Prophet, and TFT.
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from src.models.lgbm_forecaster import LGBMForecaster
from src.utils.logger import get_logger

logger = get_logger("model_registry")

class ModelRegistry:
    """Orchestrates model selection, training, persistence, and inference loading."""

    def __init__(self, registry_dir: str = "data/outputs/models"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "registry.json"
        self._init_registry_metadata()

    def _init_registry_metadata(self):
        if not self.metadata_file.exists():
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump({"models": {}, "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    def register_model_run(
        self,
        model_name: str,
        store_id: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        artifact_path: str
    ):
        """Records a completed model run and its validation benchmarks."""
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        key = f"{model_name}_{store_id}"
        data["models"][key] = {
            "model_name": model_name,
            "store_id": store_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "artifact_path": str(artifact_path)
        }
        data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Registered model artifact in registry: {key}")

    def list_models(self) -> Dict[str, Any]:
        """Returns all registered models and their recorded validation scores."""
        if not self.metadata_file.exists():
            return {}
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            return json.load(f).get("models", {})

    def load_lgbm_model(self, store_id: str) -> Optional[LGBMForecaster]:
        """Loads the active serialized LightGBM model for a store."""
        model_dir = self.registry_dir / f"lgbm_{store_id}"
        if not model_dir.exists():
            logger.warning(f"No saved LightGBM model found at {model_dir}")
            return None
        return LGBMForecaster.load_model(str(model_dir))

