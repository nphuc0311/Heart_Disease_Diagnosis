# src/config/schema.py
from pydantic import BaseModel, Field, validator
from typing import Dict, Any
from pathlib import Path

class DataConfig(BaseModel):
    raw_path: Path = Field(..., description="Path to raw data")
    processed_path: Path = Field(default=Path("data/processed"), description="Path to processed data")
    val_size: float = Field(0.2, ge=0.0, le=1.0)
    test_size: float = Field(0.1, ge=0.0, le=1.0)
    random_state: int = Field(42, ge=0)

class AppConfig(BaseModel):
    data: DataConfig

    @validator('data')
    def validate_paths(cls, v):
        if not v.raw_path.exists():
            raise ValueError(f"Raw data path does not exist: {v.raw_path}")
        return v