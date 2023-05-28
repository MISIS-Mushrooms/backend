import json

import pandas as pd
from faker import Faker

from dedformer.model import UserBank

if __name__ == '__main__':
    fake = Faker("ru_RU")
    users = UserBank('data/users.csv')
    user_names = {idx: {
        'first': fake.first_name_male() if is_male else fake.first_name_female(),
        'last': fake.last_name_male() if is_male else fake.last_name_female(),
        'middle': fake.middle_name_male() if is_male else fake.middle_name_female()
    } for idx, is_male in users.get_ids_genders()}
    user_names = pd.DataFrame.from_dict(user_names, orient='index')
    user_names.index.name = 'index'
    user_names.to_csv('data/user_names.csv')
