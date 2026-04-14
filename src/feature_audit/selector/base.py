from abc import ABC, abstractmethod
import pandas as pd


class FeatureSelectionStrategy(ABC):
    @abstractmethod
    def select(self) -> pd.DataFrame:
        """
        Возвращает DataFrame минимум с колонкой 'feature'.
        """
        raise NotImplementedError