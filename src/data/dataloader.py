# src/config/loader.py

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from src.utils.seed import set_seed


# Set seeds for reproducibility
set_seed()


class HeartDiseaseDataLoader:
    """
    A data loader class for the Heart Disease Diagnosis project.
    Handles loading raw data, preprocessing, splitting, feature engineering,
    and feature selection to generate various dataset variants (raw, DT-selected, FE, FE+DT).
    
    Assumes raw data is in 'data/raw/heart_disease_full.csv' or a provided path.
    Outputs processed splits to 'data/processed/' by default.
    """

    COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

    NUMERIC_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    CATEGORICAL_COLS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    GEN_NUM = ['chol_per_age', 'bps_per_age', 'hr_ratio']
    GEN_CAT = ['age_bin']
    ALL_NUMS = [c for c in NUMERIC_COLS] + GEN_NUM
    ALL_CATS = [c for c in CATEGORICAL_COLS] + GEN_CAT

    TARGET = 'target'
    K_FEATURES_DEFAULT = 10
    K_MI_DEFAULT = len(COLUMNS) - 1  # All original features for MI selection

    def __init__(
        self,
        data_path: str = 'data/raw/heart_disease_full.csv',
        processed_dir: str = 'data/processed',
        k_features: int = K_FEATURES_DEFAULT,
        k_mi: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Initialize the data loader.
        
        Args:
            data_path (str): Path to the raw CSV file.
            processed_dir (str): Directory to save processed splits.
            k_features (int): Number of top features to select via DT.
            k_mi (int, optional): Number of top features to select via MI (defaults to original feature count).
            random_state (int): Random seed for splits.
        """
        self.data_path = Path(data_path)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.k_features = k_features
        self.k_mi = k_mi or self.K_MI_DEFAULT
        self.random_state = random_state
        
        # Internal storage for datasets
        self.raw_data = None
        self.X_train_raw = None
        self.X_val_raw = None
        self.X_test_raw = None
        self.y_train_raw = None
        self.y_val_raw = None
        self.y_test_raw = None
        
        self.X_train_dt = None
        self.X_val_dt = None
        self.X_test_dt = None
        
        self.X_train_fe = None
        self.X_val_fe = None
        self.X_test_fe = None
        
        self.X_train_fe_dt = None
        self.X_val_fe_dt = None
        self.X_test_fe_dt = None
        
        self.feature_names_raw = None
        self.feature_names_fe = None
        self.selected_features_dt = None
        self.selected_features_mi = None

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load and preprocess the raw CSV data.
        Sets columns, converts types, handles missing values, and binarizes target.
        
        Returns:
            pd.DataFrame: The raw processed DataFrame.
        """
        if self.raw_data is not None:
            return self.raw_data
        
        raw = pd.read_csv(self.data_path, header=None)
        raw.columns = self.COLUMNS
        
        # Convert specified columns to numeric, coercing errors to NaN
        for c in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')
        
        # Binarize target: >0 -> 1, else 0
        raw[self.TARGET] = (raw[self.TARGET] > 0).astype(int)
        
        print(f"Raw data shape: {raw.shape}")
        print("Missing values:\n", raw.isna().sum())
        
        self.raw_data = raw
        return raw

    def _create_preprocessing_pipeline(self, use_minmax_for_cat: bool = True) -> Pipeline:
        """
        Create the raw preprocessing pipeline.
        
        Args:
            use_minmax_for_cat (bool): Use MinMaxScaler for categorical (as in raw) vs OneHot for FE.
        
        Returns:
            Pipeline: The preprocessing pipeline.
        """
        cat_proc = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', MinMaxScaler() if use_minmax_for_cat else OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        num_proc = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocess = ColumnTransformer([
            ('num', num_proc, self.NUMERIC_COLS),
            ('cat', cat_proc, self.CATEGORICAL_COLS),
        ])
        
        return Pipeline([('preprocess', preprocess)])

    def create_splits(self, test_size: float = 0.2, val_size: float = 0.5) -> None:
        """
        Split the raw data into train/val/test sets after preprocessing.
        
        Args:
            test_size (float): Initial test split size.
            val_size (float): Val split size from temp set.
        """
        if self.raw_data is None:
            self.load_raw_data()
        
        raw_feature_cols = [c for c in self.raw_data.columns if c != self.TARGET]
        X_all = self.raw_data[raw_feature_cols]
        y_all = self.raw_data[self.TARGET]
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_all, y_all, test_size=test_size, stratify=y_all, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=self.random_state
        )
        
        # Preprocess
        raw_pipeline = self._create_preprocessing_pipeline(use_minmax_for_cat=True)
        raw_pipeline.fit(X_train, y_train)
        
        self.X_train_raw = raw_pipeline.transform(X_train)
        self.X_val_raw = raw_pipeline.transform(X_val)
        self.X_test_raw = raw_pipeline.transform(X_test)
        
        self.y_train_raw = y_train
        self.y_val_raw = y_val
        self.y_test_raw = y_test
        
        # Get feature names
        preprocess = raw_pipeline.named_steps['preprocess']
        self.feature_names_raw = []
        for name, transformer, columns in preprocess.transformers_:
            if hasattr(transformer, 'get_feature_names_out'):
                self.feature_names_raw.extend(transformer.get_feature_names_out(columns))
            else:
                self.feature_names_raw.extend(columns)
        
        print(f"Splits created: Train {self.X_train_raw.shape}, Val {self.X_val_raw.shape}, Test {self.X_test_raw.shape}")

    def _dt_feature_selection(self) -> None:
        """
        Perform Decision Tree-based feature selection on raw preprocessed data.
        Selects top K features based on importance.
        """
        if self.X_train_raw is None:
            self.create_splits()
        
        dt_pipeline = Pipeline([
            ('preprocess', self._create_preprocessing_pipeline(use_minmax_for_cat=True)),
            ('decision_tree', DecisionTreeClassifier(random_state=self.random_state))
        ])
        
        # Fit on train (but since splits are already preprocessed, we refit preprocess if needed)
        # For selection, fit DT on preprocessed train
        dt_clf = DecisionTreeClassifier(random_state=self.random_state)
        dt_clf.fit(self.X_train_raw, self.y_train_raw)
        
        feature_importances = pd.Series(
            dt_clf.feature_importances_, index=self.feature_names_raw
        ).sort_values(ascending=False)
        
        print("Sorted Feature Importances:\n", feature_importances.head(self.k_features))
        self.selected_features_dt = feature_importances.head(self.k_features).index.tolist()
        
        # Apply selection
        self.X_train_dt = pd.DataFrame(self.X_train_raw, columns=self.feature_names_raw, index=self.y_train_raw.index)[self.selected_features_dt]
        self.X_val_dt = pd.DataFrame(self.X_val_raw, columns=self.feature_names_raw, index=self.y_val_raw.index)[self.selected_features_dt]
        self.X_test_dt = pd.DataFrame(self.X_test_raw, columns=self.feature_names_raw, index=self.y_test_raw.index)[self.selected_features_dt]
        
        print(f"DT-selected features: {self.selected_features_dt}")
        print(f"DT shapes: Train {self.X_train_dt.shape}, Val {self.X_val_dt.shape}, Test {self.X_test_dt.shape}")

    def _add_new_features_transformer(self) -> 'AddNewFeaturesTransformer':
        """Instantiate the custom feature engineering transformer."""
        return AddNewFeaturesTransformer()

    def _create_fe_preprocessing_pipeline(self) -> Pipeline:
        """
        Create the FE preprocessing pipeline with added features and OHE for cats.
        """
        num_proc = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
        cat_proc = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), 
                             ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        
        pre = ColumnTransformer([
            ('num', num_proc, self.ALL_NUMS),
            ('cat', cat_proc, self.ALL_CATS),
        ], verbose_feature_names_out=False).set_output(transform='pandas')
        
        add_transformer = self._add_new_features_transformer()
        return Pipeline([('add', add_transformer), ('pre', pre)]).set_output(transform='pandas')

    def _feature_engineering(self) -> None:
        """
        Apply feature engineering: add new features, preprocess with OHE, and select via MI.
        """
        if self.raw_data is None:
            self.load_raw_data()
        
        raw_feature_cols = [c for c in self.raw_data.columns if c != self.TARGET]
        X_train_full = self.raw_data.loc[self.y_train_raw.index, raw_feature_cols]
        X_val_full = self.raw_data.loc[self.y_val_raw.index, raw_feature_cols]
        X_test_full = self.raw_data.loc[self.y_test_raw.index, raw_feature_cols]
        
        fe_pre = self._create_fe_preprocessing_pipeline()
        Xt_tr = fe_pre.fit_transform(X_train_full, self.y_train_raw)
        Xt_va = fe_pre.transform(X_val_full)
        Xt_te = fe_pre.transform(X_test_full)
        
        # Remove zero-variance columns
        nz_cols = Xt_tr.columns[Xt_tr.nunique(dropna=False) > 1]
        Xt_tr = Xt_tr[nz_cols]
        Xt_va = Xt_va[nz_cols]
        Xt_te = Xt_te[nz_cols]
        
        # Compute MI
        ohe = fe_pre.named_steps['pre'].named_transformers_['cat'].named_steps['ohe']
        cat_names = list(ohe.get_feature_names_out(self.ALL_CATS))
        is_discrete = np.array([c in cat_names for c in Xt_tr.columns], dtype=bool)
        
        mi = mutual_info_classif(
            Xt_tr.values, self.y_train_raw.values,
            discrete_features=is_discrete, random_state=self.random_state
        )
        mi_series = pd.Series(mi, index=Xt_tr.columns).sort_values(ascending=False)
        
        self.selected_features_mi = list(mi_series.head(self.k_mi).index)
        self.feature_names_fe = self.selected_features_mi
        
        # Apply selection
        self.X_train_fe = Xt_tr[self.selected_features_mi]
        self.X_val_fe = Xt_va[self.selected_features_mi]
        self.X_test_fe = Xt_te[self.selected_features_mi]
        
        print(f"MI-selected features: {self.selected_features_mi[:10]}...")  # Top 10 for brevity
        print(f"FE shapes: Train {self.X_train_fe.shape}, Val {self.X_val_fe.shape}, Test {self.X_test_fe.shape}")

    def _fe_dt_selection(self) -> None:
        """
        Apply FE and then DT selection on top of it.
        """
        # First, get FE datasets
        self._feature_engineering()
        
        # Then, apply DT selection on FE features
        dt_clf = DecisionTreeClassifier(random_state=self.random_state)
        dt_clf.fit(self.X_train_fe, self.y_train_raw)
        
        feature_importances_fe = pd.Series(
            dt_clf.feature_importances_, index=self.feature_names_fe
        ).sort_values(ascending=False)
        
        fe_dt_features = feature_importances_fe.head(self.k_features).index.tolist()
        
        self.X_train_fe_dt = self.X_train_fe[fe_dt_features]
        self.X_val_fe_dt = self.X_val_fe[fe_dt_features]
        self.X_test_fe_dt = self.X_test_fe[fe_dt_features]
        
        self.selected_features_dt_fe = fe_dt_features  # For reference
        
        print(f"FE+DT selected features: {fe_dt_features}")
        print(f"FE+DT shapes: Train {self.X_train_fe_dt.shape}, Val {self.X_val_fe_dt.shape}, Test {self.X_test_fe_dt.shape}")

    def prepare_all_datasets(self, save_to_csv: bool = True) -> Dict[str, Dict[str, Tuple[pd.DataFrame, pd.Series]]]:
        """
        Prepare all dataset variants: raw, DT, FE, FE+DT.
        
        Args:
            save_to_csv (bool): Whether to save splits as CSV files.
        
        Returns:
            Dict: {
                'raw': {'train': (X_train, y_train), 'val': ..., 'test': ...},
                'dt': ...,
                'fe': ...,
                'fe_dt': ...
            }
        """
        self.create_splits()
        
        # DT on raw
        self._dt_feature_selection()
        
        # FE on raw
        self._feature_engineering()
        
        # FE + DT
        self._fe_dt_selection()
        
        datasets = {
            'raw': self._get_split_dict(self.X_train_raw, self.X_val_raw, self.X_test_raw,
                                        self.y_train_raw, self.y_val_raw, self.y_test_raw,
                                        self.feature_names_raw),
            'dt': self._get_split_dict(self.X_train_dt, self.X_val_dt, self.X_test_dt,
                                       self.y_train_raw, self.y_val_raw, self.y_test_raw,
                                       self.selected_features_dt),
            'fe': self._get_split_dict(self.X_train_fe, self.X_val_fe, self.X_test_fe,
                                       self.y_train_raw, self.y_val_raw, self.y_test_raw,
                                       self.feature_names_fe),
            'fe_dt': self._get_split_dict(self.X_train_fe_dt, self.X_val_fe_dt, self.X_test_fe_dt,
                                          self.y_train_raw, self.y_val_raw, self.y_test_raw,
                                          self.selected_features_dt_fe)
        }
        
        if save_to_csv:
            self._save_datasets(datasets)
        
        return datasets

    def _get_split_dict(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        feature_names: list
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Helper to create split dict with DataFrames."""
        return {
            'train': (
                pd.DataFrame(X_train, columns=feature_names, index=y_train.index),
                y_train
            ),
            'val': (
                pd.DataFrame(X_val, columns=feature_names, index=y_val.index),
                y_val
            ),
            'test': (
                pd.DataFrame(X_test, columns=feature_names, index=y_test.index),
                y_test
            )
        }

    def _save_datasets(self, datasets: Dict) -> None:
        """Save all datasets as CSV files."""
        for mode, splits in datasets.items():
            for split_name, (X, y) in splits.items():
                df = pd.concat([X, y.rename(self.TARGET)], axis=1)
                df.to_csv(self.processed_dir / f'{mode}_{split_name}.csv', index=False)
                print(f"Saved: {mode}_{split_name}.csv")

    def get_dataset(
        self,
        mode: str = 'raw',
        split: str = 'train'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get a specific dataset split.
        
        Args:
            mode (str): 'raw', 'dt', 'fe', 'fe_dt'.
            split (str): 'train', 'val', 'test'.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X, y)
        """
        if self.X_train_raw is None:
            self.prepare_all_datasets(save_to_csv=False)
        
        if mode not in ['raw', 'dt', 'fe', 'fe_dt']:
            raise ValueError(f"Mode must be one of: raw, dt, fe, fe_dt")
        
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Split must be one of: train, val, test")
        
        key = f"X_{split}_{mode.replace('+', '_')}"
        X = getattr(self, key, None)
        y_key = f"y_{split}_raw"  # y is shared
        y = getattr(self, y_key, None)
        
        if X is None or y is None:
            raise ValueError(f"Dataset {mode}/{split} not prepared. Run prepare_all_datasets() first.")
        
        # Convert to DF if needed
        if not isinstance(X, pd.DataFrame):
            if mode == 'raw':
                X = pd.DataFrame(X, columns=self.feature_names_raw)
            elif mode == 'dt':
                X = pd.DataFrame(X, columns=self.selected_features_dt)
            elif mode == 'fe':
                X = pd.DataFrame(X, columns=self.feature_names_fe)
            elif mode == 'fe_dt':
                X = pd.DataFrame(X, columns=self.selected_features_dt_fe)
        
        return X, y

    def plot_mi_scores(self, top_n: int = 20) -> None:
        """
        Plot top MI scores (requires FE to be prepared).
        """
        if self.X_train_fe is None:
            self._feature_engineering()
        
        # Assuming mi_series is computed in _feature_engineering, but for plot, recompute if needed
        # (In practice, store mi_series in class)
        fe_pre = self._create_fe_preprocessing_pipeline()
        Xt_tr = fe_pre.fit_transform(
            self.raw_data.loc[self.y_train_raw.index, [c for c in self.raw_data.columns if c != self.TARGET]],
            self.y_train_raw
        )
        nz_cols = Xt_tr.columns[Xt_tr.nunique(dropna=False) > 1]
        Xt_tr = Xt_tr[nz_cols]
        
        ohe = fe_pre.named_steps['pre'].named_transformers_['cat'].named_steps['ohe']
        cat_names = list(ohe.get_feature_names_out(self.ALL_CATS))
        is_discrete = np.array([c in cat_names for c in Xt_tr.columns], dtype=bool)
        
        mi = mutual_info_classif(
            Xt_tr.values, self.y_train_raw.values,
            discrete_features=is_discrete, random_state=self.random_state
        )
        mi_series = pd.Series(mi, index=Xt_tr.columns).sort_values(ascending=False)
        
        topN = mi_series.head(top_n).iloc[::-1]
        plt.figure(figsize=(10, max(6, 0.35 * top_n)))
        plt.barh(topN.index, topN.values)
        plt.title('Top MI scores (Train)')
        plt.xlabel('MI score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(self.processed_dir / 'top_mi_scores.pdf', bbox_inches='tight')
        plt.show()


class AddNewFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add new engineered features.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.columns_ = X.columns
        self.new_features_ = []
        if {'chol', 'age'} <= set(X.columns):
            self.new_features_.append('chol_per_age')
        if {'trestbps', 'age'} <= set(X.columns):
            self.new_features_.append('bps_per_age')
        if {'thalach', 'age'} <= set(X.columns):
            self.new_features_.append('hr_ratio')
        if 'age' in X.columns:
            self.new_features_.append('age_bin')
        return self

    def transform(self, X):
        df = X.copy()
        if {'chol', 'age'} <= set(df.columns):
            df['chol_per_age'] = df['chol'] / df['age']
        if {'trestbps', 'age'} <= set(df.columns):
            df['bps_per_age'] = df['trestbps'] / df['age']
        if {'thalach', 'age'} <= set(df.columns):
            df['hr_ratio'] = df['thalach'] / df['age']
        if 'age' in df.columns:
            df['age_bin'] = pd.cut(df['age'], bins=5, labels=False).astype('category')
        return df

    def get_feature_names_out(self, input_features=None):
        return list(self.columns_) + self.new_features_