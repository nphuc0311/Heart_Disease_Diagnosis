from src.config.loader import load_config
from src.utils.seed import set_seed
from src.data.dataloader import HeartDiseaseDataLoader

config = load_config()
set_seed(config)

loader = HeartDiseaseDataLoader(
    data_path=config.data.raw_path,
    processed_dir=config.data.processed_path,
    k_features=config.data.k_features
)

loader.prepare_all_datasets(save_to_csv=True)
