import datetime as dt
import pickle

import pandas as pd
from pandarallel import pandarallel
from tqdm.auto import tqdm

from dedformer.model import GroupBank

if __name__ == '__main__':
    pandarallel.initialize(progress_bar=True)
    all_data = {}
    bank = GroupBank('data/groups.csv')
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
        group_seq = list(group_data['уникальный номер группы'])
        group_seq_macro = list([bank.get_macro_group_id(x) for x in group_seq])
        visited = set()
        group_seq_macro_unique = []
        for i in group_seq_macro:
            if i not in visited:
                visited.add(i)
                group_seq_macro_unique.append(i)
        if len(group_seq_macro_unique) == 0:
            continue
        dct = {
            'group_sequence': group_seq_macro_unique
        }
        all_data[group_name] = dct
    with open('data/attendance.pkl', 'wb') as f:
        pickle.dump(all_data, f)
    print(123)