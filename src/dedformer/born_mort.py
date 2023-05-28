import json
from typing import Dict, Union

import pandas as pd


class BornMortFeatures():
    def __init__(self,
                 source_path: str):
        self.bornstat_df = pd.read_csv(source_path / 'bornstat.csv', index_col='Year')
        self.babybooms_df = pd.read_csv(source_path / 'babybooms.csv',
                                        index_col='year')
        self.deathcause_df_dict = {gender:
                                       pd.read_csv(source_path / 'deathcause' / f'{gender}.csv',
                                                   index_col=0,
                                                   names=[f'deathcause_{x}' for x in [47, 93, 99, 159]])
                                   for gender in
                                   ['male', 'female']
                                   }

    def shorten_range(self,
                      num: int,
                      start: int,
                      end: int) -> int:
        return min(end, max(start, num))

    def get_born_data(self,
                      born_year: int) -> Dict[str, float]:
        born_year = self.shorten_range(born_year, 1959, 1988)
        return self.bornstat_df.loc[born_year, :]

    def get_deathcause_data(self,
                            born_year: int,
                            gender: str) -> Dict[str, float]:
        born_year = self.shorten_range(born_year, 1959, 1988)
        return self.deathcause_df_dict[gender].loc[born_year, :]

    def get_babybooms_data(self,
                           born_year: int) -> Dict[str, float]:
        return self.babybooms_df.loc[born_year, :]

    def get_features_by_id(self,
                           born_year: int,
                           is_male: bool) -> Dict[str, float]:
        gender = 'male' if is_male else 'female'
        features = {}
        features.update(self.get_born_data(born_year))
        features.update(self.get_deathcause_data(born_year, gender))
        features.update(self.get_babybooms_data(born_year))
        return features
