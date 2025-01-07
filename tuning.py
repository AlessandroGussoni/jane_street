import optuna
from optuna.terminator.callback import TerminatorCallback
import lightgbm as lgb
import xgboost as xgb
import polars as pl
import numpy as np
from typing import Dict, List, Optional, Literal
from sklearn.decomposition import PCA
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
import joblib

from validation import PurgedCrossValidator
from feng import FeatureEngineer
from loss import (
    weighted_rmse_loss, weighted_rmse_eval, weighted_r2_score,
    weighted_rmse_objective_xgb, weighted_rmse_metric_xgb, weighted_r2_metric_xgb
)

import os

@dataclass
class OptunaOptimizer:

    def __init__(self,
                data_path: str,
                output_path: str,
                model_type: Literal["lightgbm", "xgboost"] = "lightgbm",
                n_trials: int = 3,
                n_validation_partitions: int = 1,
                purge_embargo_periods: int = 5,
                early_stopping_rounds: int = 150,
                random_state: int = 42,
                feature_cols: List[str] = field(default_factory=list),
                target_col: List[str] = "responder_6",
                weight_col: str = "weight",
                categorical_cols: List[str] = field(default_factory=list),
                gpu_id: Optional[int] = None  # None for CPU, int for GPU device ID
                ):
        
        self.data_path = data_path
        self.output_path = output_path
        self.model_type = model_type
        self.n_trials = n_trials
        self.n_validation_partitions = n_validation_partitions
        self.purge_embargo_periods = purge_embargo_periods
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.weight_col = weight_col
        self.categorical_cols = categorical_cols
        self.gpu_id = gpu_id
    
        self.validator = PurgedCrossValidator(
            n_validation_partitions=self.n_validation_partitions,
            purge_embargo_periods=self.purge_embargo_periods,
            data_path=self.data_path
        )
        
        self.feature_engineer = FeatureEngineer()
        
        self.output_dir = Path(self.output_path)
        self.models_dir = self.output_dir / "models"
        self.scores_dir = self.output_dir / "scores"
        self.artifacts_dir = self.output_dir / "artifacts" 
        
        for directory in [self.output_dir, self.models_dir, self.scores_dir, self.artifacts_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        self.trial_scores = {}
        
        self.symbol_id_counts = pl.scan_csv("/Users/alessandro.gussoni/Documents/jane_street/data/date_id_symbol.csv")
        self.original_features: List[str] = [f"feature_{str(i).zfill(2)}" for i in range(79)]

    def _suggest_lightgbm_params(self, trial: optuna.Trial, columns: List[str]) -> Dict:
        params = {
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": self.random_state,
            
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 5000),
            
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 8, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-8, 1.0, log=True),
            
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "num_threads": 16
        }
        
        if self.gpu_id is not None:
            params.update({
                "device": "gpu",
                "gpu_device_id": self.gpu_id
            })

        contrib = [1.0] * (len(columns) + 15)
        contrib[columns.index("symbol_id")] = trial.suggest_float("feature_penalty", 0.4, 0.99)
        params["feature_contrib"] = contrib
        
        return params

    def _suggest_xgboost_params(self, trial: optuna.Trial) -> Dict:
        params = {
            "verbosity": 0,
            "random_state": self.random_state,
            
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 10000),
            
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            
            # XGBoost-specific parameters
            "max_bin": trial.suggest_int("max_bin", 128, 512),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        }
        
        if self.gpu_id is not None:
            params.update({
                "tree_method": "gpu_hist",
                "gpu_id": self.gpu_id
            })
        else:
            params["tree_method"] = "hist"
        
        return params

    def _train_lightgbm_model(self, params: Dict, train_dataset: lgb.Dataset, 
                             val_dataset: lgb.Dataset) -> lgb.Booster:
        """Train a LightGBM model with given parameters"""
        params["objective"] = weighted_rmse_loss
        callbacks = [lgb.early_stopping(self.early_stopping_rounds, first_metric_only=True)]
        
        model = lgb.train(
            params,
            train_dataset,
            valid_sets=[train_dataset, val_dataset],
            feval=[weighted_r2_score],
            callbacks=callbacks,
        )
        return model

    def _train_xgboost_model(self, params: Dict, train_dataset: xgb.DMatrix, 
                            val_dataset: xgb.DMatrix) -> xgb.Booster:
        """Train an XGBoost model with given parameters"""
        params["objective"] = "reg:squarederror"#weighted_rmse_objective_xgb
        
        evals = [(val_dataset, 'valid')]
        callbacks = [
            xgb.callback.EarlyStopping(
                rounds=self.early_stopping_rounds,
                metric_name='w-r2',
                maximize=True
            )
        ]
        
        model = xgb.train(
            params,
            train_dataset,
            num_boost_round=params['n_estimators'],
            evals=evals,
            custom_metric=weighted_r2_metric_xgb,
            callbacks=callbacks
        )
        return model

    def _prepare_training_data(self, 
                               train_df: pl.LazyFrame, 
                               val_df: pl.LazyFrame, 
                               fold_idx: Optional[int] = None) -> tuple:
        feng = FeatureEngineer()
        train_df = feng.transform(train_df, is_train=True)
        val_df = feng.transform(val_df)

        def get_feature_columns(df: pl.LazyFrame) -> tuple[List[str], List[int]]:
            all_cols = df.columns
            rcols = [f"responder_{i}" for i in range(9)]
            feature_cols = [
                col for col in all_cols if col not in rcols + ["date_id"]]
            categorical_indices = [
                idx for idx, col in enumerate(feature_cols) if col in self.categorical_cols
            ]
            return feature_cols, categorical_indices
        
        feature_cols, categorical_indices = get_feature_columns(train_df)

        pca = PCA(n_components=15)
        
        train_data = train_df.select(feature_cols + [self.target_col]).collect()
        comps = pca.fit_transform(
            train_data.select(feature_cols).fill_null(0).fill_nan(0).to_numpy()
        )
        
        X_train = train_data.select(feature_cols).to_numpy()
        y_train = train_data.select(self.target_col).to_numpy().ravel()
        w_train = train_data.select(self.weight_col).to_numpy().ravel()
        
        X_train = np.c_[X_train, comps]
        
        val_data = val_df.select(feature_cols + [self.target_col]).collect()
        comps = pca.transform(
            val_data.select(feature_cols).fill_null(0).fill_nan(0).to_numpy()
        )
        X_val = val_data.select(feature_cols).to_numpy()
        y_val = val_data.select(self.target_col).to_numpy().ravel()
        w_val = val_data.select(self.weight_col).to_numpy().ravel()
        X_val = np.c_[X_val, comps]

        component_columns = [f"component_{i}" for i in range(15)]
        
        # Save fitted objects if fold_idx is provided (during final training)
        if fold_idx is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save FeatureEngineer
            feng_path = self.artifacts_dir / f"feature_engineer_fold_{fold_idx}_{timestamp}.joblib"
            joblib.dump(feng, feng_path)
            
            # Save PCA
            pca_path = self.artifacts_dir / f"pca_fold_{fold_idx}_{timestamp}.joblib"
            joblib.dump(pca, pca_path)
        
        if self.model_type == "lightgbm":
            train_dataset = lgb.Dataset(
                X_train, y_train, weight=w_train,
                categorical_feature=categorical_indices,
                feature_name=feature_cols + component_columns,
                free_raw_data=True
            )
            val_dataset = lgb.Dataset(
                X_val, y_val, weight=w_val,
                categorical_feature=categorical_indices,
                feature_name=feature_cols + component_columns,
                reference=train_dataset,
                free_raw_data=True
            )
        else:  # xgboost
            train_dataset = xgb.DMatrix(
                X_train, y_train, weight=w_train,
                feature_names=feature_cols,
                enable_categorical=True
            )
            val_dataset = xgb.DMatrix(
                X_val, y_val, weight=w_val,
                feature_names=feature_cols,
                enable_categorical=True
            )
        
        return train_dataset, val_dataset, val_df, feature_cols

    def _suggest_params(self, trial: optuna.Trial, columns: List[str]) -> Dict:
        if self.model_type == "lightgbm":
            return self._suggest_lightgbm_params(trial, columns)
        else:
            return self._suggest_xgboost_params(trial, columns)

    def _train_model(self, params: Dict, train_dataset, val_dataset):
        if self.model_type == "lightgbm":
            return self._train_lightgbm_model(params, train_dataset, val_dataset)
        else:
            return self._train_xgboost_model(params, train_dataset, val_dataset)

    def objective(self, trial: optuna.Trial) -> float:
        fold_scores = []
        
        for fold_idx, (train_df, val_df) in enumerate(self.validator.splits()):
            train_dataset, val_dataset, _, features = self._prepare_training_data(train_df, val_df, fold_idx=fold_idx)
            params = self._suggest_params(trial, features)
            model = self._train_model(params, train_dataset, val_dataset)
            
            if self.model_type == "lightgbm":
                valid_r2 = model.best_score['valid_1']['w-r2_score']
                train_r2 = model.best_score['training']['w-r2_score']
                best_iteration = model.best_iteration
            else:
                valid_r2 = model.best_score
                train_r2 = model.best_score
                best_iteration = model.best_iteration
            
            fold_metrics = {
                'valid_r2': valid_r2,
                'train_r2': train_r2,
                'best_iteration': best_iteration
            }
            
            self._save_model_and_scores(model, trial.number, fold_idx, fold_metrics)
            fold_scores.append(fold_metrics['valid_r2'])
            del train_dataset, val_dataset
            
        mean_score = np.mean(fold_scores)
        trial.set_user_attr("cv_score", mean_score)
        return mean_score

    def _save_model_and_scores(self, model, trial_number: int, 
                             fold_number: int, scores: Dict) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.model_type == "lightgbm":
            model_path = self.models_dir / f"trial_{trial_number}_fold_{fold_number}_{timestamp}.txt"
            model.save_model(str(model_path))
        else:
            model_path = self.models_dir / f"trial_{trial_number}_fold_{fold_number}_{timestamp}.json"
            model.save_model(str(model_path))
                
        self.trial_scores.setdefault(trial_number, {})[fold_number] = scores
    
    def _save_trial_scores(self) -> None:
        scores_path = self.scores_dir / f"fold_scores_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(scores_path, 'w') as f:
            json.dump(self.trial_scores, f, indent=2)
    
    def train_final_models(self, study: optuna.Study) -> None:
        best_params = study.best_params
        if self.model_type == "lightgbm":
            best_params["objective"] = "rmse"
        else:
            best_params["objective"] = "reg:squarederror"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for fold_idx, (train_df, val_df) in enumerate(self.validator.splits()):
            train_dataset, val_dataset, val_df_final, _ = self._prepare_training_data(
                train_df, val_df, fold_idx=fold_idx
            )
            model = self._train_model(best_params, train_dataset, val_dataset)
            
            if self.model_type == "lightgbm":
                model_path = self.models_dir / f"best_model_fold_{fold_idx}_{timestamp}.txt"
            else:
                model_path = self.models_dir / f"best_model_fold_{fold_idx}_{timestamp}.json"
            
            model.save_model(str(model_path))
            
            if fold_idx == self.n_validation_partitions - 1:
                val_data_path = self.output_dir / f"final_validation_data_{timestamp}.parquet"
                val_df_final.collect().write_parquet(str(val_data_path))

    def optimize(self) -> optuna.Study:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        try:
            study.optimize(
                self.objective,
                n_trials=self.n_trials,
                show_progress_bar=True,
            )
        finally:
            self._save_trial_scores()
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_path = self.output_dir / "studies" / f"optimization_study_{timestamp}.pkl"
        joblib.dump(study, study_path)
        
        # maybe delete this and retreive from folders
        self.train_final_models(study)
        
        return study
