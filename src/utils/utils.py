import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

def load_datasets_from_csv(processed_dir: str, modes: Optional[List[str]] = None) -> Dict[str, Dict[str, Tuple[pd.DataFrame, pd.Series]]]:
    """
    Load all dataset splits from CSV files in processed_dir.
    
    Args:
        processed_dir: Path to the processed directory.
        modes: Optional list of dataset modes to load (e.g., ['raw', 'fe']). 
               If None, defaults to ['raw', 'dt', 'fe', 'fe_dt'].
    
    Returns:
        Dict of datasets: {mode: {split: (X, y)}}
    """
    processed_dir = Path(processed_dir)
    if modes is None:
        modes = ['raw', 'dt', 'fe', 'fe_dt']
    
    datasets = {}
    for mode in modes:
        datasets[mode] = {}
        for split in ['train', 'val', 'test']:
            file_path = processed_dir / f'{mode}_{split}.csv'
            if not file_path.exists():
                raise FileNotFoundError(f"{file_path} not found. Run data preparation first.")
            
            df = pd.read_csv(file_path)
            X = df.drop('target', axis=1)
            y = df['target']
            datasets[mode][split] = (X, y)
            print(f"Loaded {mode}_{split}: X.shape={X.shape}, y.shape={y.shape}")
    
    return datasets


def train_and_evaluate_models(datasets: Dict[str, Dict[str, Tuple[pd.DataFrame, pd.Series]]],
                              models: Dict[str, Any],
                              output_dir: Path,
                              optimize: bool = True,
                              n_trials: int = 50) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Train and evaluate all models on all datasets.
    
    Args:
        datasets: Dict of dataset modes to splits.
        models: Dict of model names to model instances.
        output_dir: Directory to save results.
        optimize: Whether to optimize hyperparameters.
        n_trials: Number of Optuna trials for optimization.
    
    Returns:
        Dict of results: {model: {dataset: {split: acc}}}
    """
    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, model in models.items():
        results[model_name] = {}
        for dataset_mode, splits in datasets.items():
            print(f"\nTraining {model_name} on {dataset_mode} dataset...")
            
            # Train on train split
            X_train, y_train = splits['train']
            model.fit(X_train, y_train, optimize=optimize, n_trials=n_trials)
            
            # Evaluate on val and test
            val_results = model.evaluate(splits['val'][0], splits['val'][1])
            test_results = model.evaluate(splits['test'][0], splits['test'][1])
            
            results[model_name][dataset_mode] = {
                'val_acc': val_results['accuracy'],
                'test_acc': test_results['accuracy']
            }
            
    # Plot results
    plot_results(results, list(datasets.keys()), output_dir)
    
    return results


def plot_results(results: Dict[str, Dict[str, Dict[str, float]]],
                 dataset_modes: List[str],
                 output_dir: Path) -> None:
    """
    Plot bar charts comparing validation and test accuracies across datasets for each model.
    One subplot per model.
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    n_cols = 2
    n_rows = (n_models + 1) // 2  # Dynamic rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D for single row
    axes = axes.flatten()
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        
        val_accs = []
        test_accs = []
        labels = []
        for ds in dataset_modes:
            if ds in results[model_name]:
                val_accs.append(results[model_name][ds]['val_acc'])
                test_accs.append(results[model_name][ds]['test_acc'])
                labels.append(ds.replace('_', ' + '))
            # Skip if ds not in results (though should not happen)
        
        x = np.arange(len(labels))
        width = 0.35
        
        if len(val_accs) > 0:
            ax.bar(x - width/2, val_accs, width, label='Validation Accuracy', color='tab:blue', edgecolor='black')
            ax.bar(x + width/2, test_accs, width, label='Test Accuracy', color='tab:red', edgecolor='black')
            
            ax.set_ylim(0.5, 1.05)
            ax.set_ylabel('Accuracy')
            ax.set_title(model_name, fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend(loc='upper center')
            
            # Add value labels on bars
            for i, v in enumerate(val_accs):
                ax.annotate(f'{v:.3f}', xy=(x[i] - width/2, v), ha='center', va='bottom')
            for i, v in enumerate(test_accs):
                ax.annotate(f'{v:.3f}', xy=(x[i] + width/2, v), ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(model_name)
    
    # Hide extra subplots if any
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparisons.png', dpi=300, bbox_inches='tight')