from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
from typing import Tuple


class DataReader:
    def __init__(self, path: str, target_var: str, date_col: str, window_size: int) -> None:
        self.path = path
        self.target_var = target_var
        self.date_col = date_col
        self.window_size = window_size

    def read_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, index_col=self.date_col, parse_dates=[self.date_col])
        df = df[[self.target_var]]
        return df

    def fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        df_filled = data.interpolate(method="cubic")
        return df_filled

    def data_split(self, data: pd.DataFrame, split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_size = int(len(data) * split_ratio)
        train_data, test_data = data[:train_size], data[train_size:]
        return train_data, test_data

    def batch_split(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        independent_var, dependent_var = [], []
        data = data.values
        for i in range(len(data)):
            end_ix = i + self.window_size
            if end_ix > len(data) - 1:
                break
            seq_x, seq_y = data[i:end_ix], data[end_ix]
            independent_var.append(seq_x)
            dependent_var.append(seq_y)
        return np.array(independent_var), np.array(dependent_var)

    def get_path(self) -> str:
        return self.path

    def set_path(self, new_path: str) -> None:
        self.path = new_path

    def get_target_var(self) -> str:
        return self.target_var

    def set_target_var(self, new_target_var: str) -> None:
        self.target_var = new_target_var

    def get_date_col(self) -> str:
        return self.date_col

    def set_date_col(self, new_date_col: str) -> None:
        self.date_col = new_date_col

    def get_window_size(self) -> int:
        return self.window_size

    def set_window_size(self, new_window_size: int) -> None:
        self.window_size = new_window_size
