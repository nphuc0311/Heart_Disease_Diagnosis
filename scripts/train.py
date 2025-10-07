# scripts/train.py
import pandas as pd

from src.utils.utils import load_datasets_from_csv, train_and_evaluate_models
from src.config.loader import load_config
from src.data.dataloader import HeartDiseaseDataLoader
from src.models.ensemble import DecisionTreeModel, RandomForestModel, AdaBoostModel, GradientBoostingModel

from src.utils.seed import set_seed


def main():
    config = load_config()
    set_seed(config)

    processed_dir = config.data.processed_path
    output_dir = config.paths.outputs                                                                            
    n_trials = config.training.n_trials
    
    datasets = load_datasets_from_csv(processed_dir)
    
    # Initialize models
    models = {
        'DecisionTree': DecisionTreeModel(),
        'RandomForest': RandomForestModel(),
        'AdaBoost': AdaBoostModel(),
        'GradientBoosting': GradientBoostingModel()
    }
    
    # Train and evaluate
    results = train_and_evaluate_models(datasets, models, output_dir, n_trials=n_trials)
    
    # Save results summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_dir / 'training_summary.csv')
    print("\nTraining completed. Summary saved to outputs/training_summary.csv")


if __name__ == "__main__":
    main()