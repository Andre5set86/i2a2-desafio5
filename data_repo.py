import pandas as pd
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataRepo:
    df: pd.DataFrame
    path: Path

    @staticmethod
    def load(csv_path: str) -> "DataRepo":
        p = Path(csv_path)
        df = pd.read_csv(p)
        return DataRepo(df=df, path=p)