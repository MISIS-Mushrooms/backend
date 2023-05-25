import datetime as dt
import pickle

import pandas as pd
from pandarallel import pandarallel
from tqdm.auto import tqdm

if __name__ == '__main__':
    pandarallel.initialize(progress_bar=True)
    all_data = {}
    att = pd.read_csv('data/attend.csv', dtype={
        'уникальный номер занятия': str,
        'уникальный номер группы': str,
        'уникальный номер участника': str,
        'дата занятия': str,
        'время начала занятия': str,
        'время окончания занятия': str
    })
    att['dt'] = att.parallel_apply(lambda x: dt.datetime.fromisoformat(f'{x["дата занятия"]}T{x["время начала занятия"]}'), axis=1)
    for group_name, group_data in tqdm(att.groupby('уникальный номер участника')):
        group_data = group_data.sort_values(by='dt')
        dct = {
            'group_sequence': list(group_data['уникальный номер группы'])
        }
        all_data[group_name] = dct
    with open('data/attendance.pkl', 'wb') as f:
        pickle.dump(all_data, f)
    print(123)