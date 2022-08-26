from typing import Dict
from dask import dataframe as dd


class Engine(object):
    def __init__(self, tables: Dict[str, dd.DataFrame] = {}):
        super().__init__()
        self.tables = tables

    def registerTable(self, name: str, data: dd.DataFrame) -> None:
        self.tables[name] = data

    def getTable(self, name: str) -> dd.DataFrame:
        return self.tables[name]

    