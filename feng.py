import polars as pl
import numpy as np
from typing import List, Optional, Union
from sklearn.preprocessing import QuantileTransformer

# TODO: use corr features also for time_id agg
# TODO: add time_id to quantile features

class FeatureEngineer:
    def __init__(
        self,
        lag_windows: List[int] = [1],
        hours_in_day: int = 24,
        market_open_threshold: int = 100,
        market_close_threshold: int = 700,
        symbol_mask_frac: float = 0.1
    ):
        self.lag_windows = lag_windows
        self.hours_in_day = hours_in_day
        self.market_open_threshold = market_open_threshold
        self.market_close_threshold = market_close_threshold
        self.symbol_mask_frac = symbol_mask_frac
        self.feature_cols = [f"feature_{str(i).zfill(2)}" for i in range(79)]
        self.responder_cols = [f"responder_{str(i)}" for i in range(9)]
        self.corr_matrix = None
        self.top_corr_features = None
        self.qt = QuantileTransformer(n_quantiles=25, output_distribution='uniform')
        self.train_symbols = None

        self.corr_cols = [
            "feature_06", "feature_07", "feature_05", "feature_58",
        ]

    def _find_top_correlated_features(self, df: Union[pl.LazyFrame, pl.DataFrame]) -> None:
        df = df.collect() if isinstance(df, pl.LazyFrame) else df
        
        corrs = []
        for feat in self.feature_cols:
            corr = df.select(pl.corr(feat, "responder_6")).to_numpy()[0][0]
            corrs.append((feat, corr))
        
        # put this into config i guess
        self.top_corr_features = [feat for feat, _ in sorted(corrs, key=lambda x: abs(x[1]), reverse=True)[:10]]

    def _apply_quantile_transform(self, df: Union[pl.LazyFrame, pl.DataFrame]) -> List[pl.Expr]:
        if self.top_corr_features is None:
            return []

        df = df.collect() if isinstance(df, pl.LazyFrame) else df
        
        feature_matrix = df.select(self.top_corr_features).to_numpy()
        transformed = self.qt.fit_transform(feature_matrix)
        
        return [
            pl.lit(transformed[:, i]).alias(f"{feat}_qt")
            for i, feat in enumerate(self.top_corr_features)
        ]

    def _compute_symbol_correlations(self, df: Union[pl.LazyFrame, pl.DataFrame]) -> None:
        df = df.collect() if isinstance(df, pl.LazyFrame) else df
        
        df_corr = df.with_columns([
            pl.concat_str([
                pl.col("date_id").cast(pl.Utf8),
                pl.col("time_id").cast(pl.Utf8)
            ]).alias("key")
        ])

        pivot_df = df_corr.select(["key", "symbol_id", "responder_6"]).pivot(
            index="key",
            columns="symbol_id",
            values="responder_6",
            aggregate_function="mean"
        )

        self.corr_matrix = (pivot_df
                            .fill_null(0)
                            .select(pl.exclude("key"))
                            .corr()
                            .select([pl.all().abs()])
        )

    def _mask_symbols(self, df: Union[pl.LazyFrame, pl.DataFrame], is_train: bool) -> pl.LazyFrame:
        df = df.collect() if isinstance(df, pl.LazyFrame) else df
        working_df = df.lazy()

        if is_train:
            unique_pairs = df.select(['date_id', 'symbol_id']).unique()
            n_pairs = len(unique_pairs)
            n_mask = int(n_pairs * self.symbol_mask_frac)

            np.random.seed(42)  #  this should come from config
            mask_indices = np.random.choice(n_pairs, n_mask, replace=False)
            mask_pairs = unique_pairs.to_numpy()
            mask_pairs = mask_pairs[mask_indices]
            mask_pairs = pl.DataFrame(mask_pairs, schema=['date_id', 'symbol_id'])
   
            self.train_symbols = df.select('symbol_id').unique()
            
            
            working_df = working_df.with_columns([
                pl.when(
                    pl.struct(['date_id', 'symbol_id']).is_in(mask_pairs)
                )
                .then(pl.lit(-1))
                .otherwise(pl.col("symbol_id"))
                .alias("symbol_id")
            ])
        else:
            if self.train_symbols is not None:
                working_df = working_df.with_columns([
                    pl.when(
                        ~pl.col("symbol_id").is_in(self.train_symbols.select('symbol_id'))
                    )
                    # lightgbm set this to nan
                    .then(pl.lit(-1))
                    .otherwise(pl.col("symbol_id"))
                    .alias("symbol_id")
                ])

        return working_df

    def _create_distribution_stats(self) -> List[pl.Expr]:
        stats_exprs = []
        
        for feat in self.corr_cols:
            stats_exprs.extend([
                pl.col(feat).max().over(['date_id', 'time_id']).alias(f"{feat}_max"),
                pl.col(feat).min().over(['date_id', 'time_id']).alias(f"{feat}_min"),
                pl.col(feat).quantile(0.05).over(['date_id', 'time_id']).alias(f"{feat}_q05"),
                pl.col(feat).quantile(0.95).over(['date_id', 'time_id']).alias(f"{feat}_q95"),
            ])
        
        return stats_exprs
    
    def _create_row_stats_features(self, lag_columns) -> List[pl.Expr]:
        stats_exprs = []
        num_cols = [
            col for col in self.feature_cols if col not in ["feature_09", "feature_10", "feature_11"] # cat cols
        ]
        
        stats_exprs.extend([
            pl.mean_horizontal(num_cols).alias("features_row_mean"),
            pl.min_horizontal(num_cols).alias("features_row_min"),
            pl.max_horizontal(num_cols).alias("features_row_max"),
        ])
        
        stats_exprs.extend([
            pl.mean_horizontal(lag_columns).alias("responders_row_mean"),
            pl.min_horizontal(lag_columns).alias("responders_row_min"),
            pl.max_horizontal(lag_columns).alias("responders_row_max"),
        ])
        
        return stats_exprs
    
    def _create_periodic_features(self) -> List[pl.Expr]:
        return [
            (2 * np.pi * pl.col("time_id") / self.hours_in_day).sin().alias("time_id_sin"),
            (2 * np.pi * pl.col("time_id") / self.hours_in_day).cos().alias("time_id_cos")
        ]
    
    def _create_lag_features(self, df: Union[pl.LazyFrame, pl.DataFrame], 
                           lags: Optional[pl.DataFrame] = None) -> Union[pl.LazyFrame, pl.DataFrame]:
        if lags is None:
            lag_exprs = []
            for resp in self.responder_cols:
                for lag in self.lag_windows:
                    lag_exprs.append(
                        pl.col(resp)
                        .shift(lag)
                        .over(['symbol_id', 'time_id'])
                        .alias(f"{resp}_lag_{lag}")
                    )
            return df.with_columns(lag_exprs)
        else:
            join_keys = ["symbol_id", "time_id"]
            lags_lazy = lags.lazy()
            result = df.join(
                lags_lazy,
                on=join_keys,
                how="left"
            )
            return result

    def _create_market_session_features(self) -> List[pl.Expr]:
        return [
            (
                (pl.col("time_id") < self.market_open_threshold) | 
                (pl.col("time_id") > self.market_close_threshold)
            ).alias("is_market_edge")
        ]
    
    def _create_symbol_presence_features(self, symbols_by_date: pl.DataFrame) -> List[pl.Expr]:
        presence_df = (
            symbols_by_date
            .with_columns([
                pl.col("date_id").alias("prev_date_id"),
                pl.lit(1).alias("was_present")
            ])
        )

        presence_df = presence_df.with_columns([
            (pl.col("prev_date_id") + 1).alias("date_id")
        ])
        
        return presence_df.select([
            "symbol_id",
            "date_id",
            "was_present"
        ])

    def transform(
        self, 
        df: Union[pl.LazyFrame, pl.DataFrame],
        is_train: bool = False,
        symbols_by_date: Optional[pl.DataFrame] = None,
        lags: Optional[pl.DataFrame] = None
    ) -> Union[pl.LazyFrame, pl.DataFrame]:
        is_lazy = isinstance(df, pl.LazyFrame)
        working_df = df if is_lazy else df.lazy()

        working_df = self._mask_symbols(working_df, is_train)

        if is_train:
            self._find_top_correlated_features(working_df)
            
        working_df = working_df.with_columns(self._create_distribution_stats())

        if self.top_corr_features is not None:
            working_df = working_df.with_columns(self._apply_quantile_transform(working_df))

        # if symbols_by_date is not None:
        #     presence_features = self._create_symbol_presence_features(symbols_by_date)
        #     working_df = working_df.join(
        #         presence_features,
        #         on=["symbol_id", "date_id"],
        #         how="left"
        #     ).with_columns([
        #         pl.col("was_present").fill_null(0).alias("was_present")
        #     ])
        
        working_df = self._create_lag_features(working_df, lags)

        if lags is not None:
            lag_columns = [col for col in working_df.columns if any(
                f"{resp}_lag_{lag}" in col 
                for resp in self.responder_cols 
                for lag in self.lag_windows
            )]
        else:
            lag_columns = [f"{resp}_lag_{lag}" 
                        for resp in self.responder_cols 
                        for lag in self.lag_windows]
        
        working_df = working_df.with_columns(self._create_row_stats_features(lag_columns))
        working_df = working_df.with_columns(self._create_periodic_features())
    
        return working_df if is_lazy else working_df.collect()
