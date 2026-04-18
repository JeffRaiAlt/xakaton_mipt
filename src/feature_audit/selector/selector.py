from typing import Sequence
import pandas as pd

from .base import FeatureSelectionStrategy


class FeatureSelector:
    def __init__(self, strategy: FeatureSelectionStrategy) -> None:
        self.strategy = strategy

    def select(self) -> pd.DataFrame:
        return self.strategy.select()