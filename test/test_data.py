from src.config.loader import config
from src.data.dataloader import HeartDiseaseDataLoader

if __name__ == "__main__":
    data_loader = HeartDiseaseDataLoader(
        data_path=config.data.raw_path,
        processed_dir=config.data.processed_path,
        k_features=10,
        random_state=config.data.random_state
    )

    datasets = data_loader.prepare_all_datasets(save_to_csv=True)
    

