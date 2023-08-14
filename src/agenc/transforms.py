import polars as pl
from abc import ABC, abstractmethod


class Transform(ABC):
    @abstractmethod
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        ...


class StandardScaler(Transform):
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.select((pl.all() - pl.all().mean()) / pl.all().std())
