# src/config/schema.py
from pydantic import BaseModel, Field, validator
from pathlib import Path

class DataConfig(BaseModel):
    raw_path: Path = Field(..., description="Path to raw data")
    processed_path: Path = Field(default=Path("data/processed"), description="Path to processed data")
    val_size: float = Field(0.2, ge=0.0, le=1.0)
    test_size: float = Field(0.1, ge=0.0, le=1.0)
    k_features: int = Field(10, ge=1, description="Number of top features to select")
    random_state: int = Field(42, ge=0)

class PathConfig(BaseModel):
    outputs: Path = Field(default=Path("outputs"), description="Directory to save outputs")
    models: Path = Field(default=Path("models"), description="Directory to save models")

class TrainingConfig(BaseModel):
    n_trials: int = Field(50, ge=1, description="Number of trials for hyperparameter optimization")

class AppConfig(BaseModel):
    data: DataConfig
    paths: PathConfig
    training: TrainingConfig

    @validator('data')
    def validate_paths(cls, v):
        if not v.raw_path.exists():
            raise ValueError(f"Raw data path does not exist: {v.raw_path}")
        return v