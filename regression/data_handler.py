import pandas as pd
from pandas import DataFrame


class DataLoader:
    def __init__(self, file_path: str):
        self._file_path = file_path

    def load_csv(self) -> DataFrame:
        return pd.read_csv(self._file_path)
