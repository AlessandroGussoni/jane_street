import pandas as pd
import polars as pl

from tuning import OptunaOptimizer
import json
import os
import warnings

# TODO: path in config are absolute buuuuu


def main_fn(config):
      warnings.filterwarnings("ignore")
      optimizer = OptunaOptimizer(
                **config
        )
      study = optimizer.optimize()

if __name__ == "__main__":
     with open("config.json", "rb") as file:
          config = json.load(file)
     main_fn(config)