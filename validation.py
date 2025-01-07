import polars as pl
from typing import Iterator, Tuple, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import os


class PurgedCrossValidator:

    def __init__(self, 
                 n_validation_partitions: int,
                 purge_embargo_periods: int,
                 data_path: str,
                 addiditona_embargo: int= 10,
                 file_idx: list = [str(i) for i in range(10)],
                 host: Optional[str] = "local",
                 date_col: Optional[str] = "date_id"):
        self.n_validation_partitions = n_validation_partitions
        self.purge_embargo_periods = purge_embargo_periods
        self.data_path = data_path
        self.addiditona_embargo = addiditona_embargo
        self.file_idx = file_idx
        self.host = host
        self.date_col = date_col
                    
    def splits(self) -> Iterator[Tuple[pl.LazyFrame, pl.LazyFrame]]:
        if self.host == "local":
            parquet_files = sorted(os.listdir(self.data_path))
            parquet_files = [f for f in parquet_files if f.split(".")[0].split("-")[-1] in self.file_idx]
            parquet_files = [
                os.path.join(self.data_path, f) for f in parquet_files
            ]
        elif self.host == "kaggle":
            parquet_files = [
                os.path.join(self.data_path, f"partition_id={i}", "part-0.parquet") for i in self.file_idx
            ]
        n_partitions = len(parquet_files)
        n_folds = n_partitions // self.n_validation_partitions
        for fold in range(n_folds):
            val_start_idx = fold * self.n_validation_partitions
            val_end_idx = val_start_idx + self.n_validation_partitions
            val_files = parquet_files[val_start_idx:val_end_idx]
            train_files = [f for f in parquet_files if f not in val_files]
            
            val_df = pl.concat([pl.scan_parquet(str(f)) for f in val_files])
            val_times = (val_df.select(pl.col(self.date_col))
                        .unique()
                        .collect()
                        .to_series()
                        .sort())
            
            min_val_time = val_times.min()
            max_val_time = val_times.max()
            train_df = pl.concat([pl.scan_parquet(str(f)) for f in train_files])
            train_df = train_df.filter(
                (pl.col(self.date_col) < min_val_time - self.purge_embargo_periods) |
                (pl.col(self.date_col) > max_val_time + self.purge_embargo_periods + self.addiditona_embargo)
            )
            
            yield train_df, val_df