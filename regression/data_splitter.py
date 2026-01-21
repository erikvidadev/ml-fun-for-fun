from typing import Optional, List, Tuple
from pandas import DataFrame, Series

from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(
            self,
            target_column: str,
            feature_columns: Optional[List[str]] = None,
            test_size: float = 0.2,
            random_state: int = 42
    ):
        self._feature_columns = feature_columns
        self._target_column = target_column
        self._test_size = test_size
        self._random_state = random_state

    def split(self, data: DataFrame) -> Tuple[DataFrame, DataFrame, Series, Series]:
        if self._feature_columns is None:
            X = data.drop(columns=[self._target_column])
        else:
            X = data[self._feature_columns]

        y = data[self._target_column]

        return train_test_split(
            X, y, test_size=self._test_size, random_state=self._random_state
        )