import pickle

import pandas as pd
import datetime as dt
import torch
from tqdm.auto import tqdm

from dedformer.model import GroupBank, UserBank, AllVectorizer, AttendanceCollator, FeatureCreator

def _get_match_rate(time_list, timetable):
    rate = 0
    if timetable is None:
        return 0
    for time_dow, time_time in time_list:
        time_at_tt = timetable[time_dow]
        if time_at_tt != 'нет':
            time_at_tt_from, time_at_tt_to = time_at_tt.split('-')
            time_at_tt_from, time_at_tt_to = dt.time(*map(int, time_at_tt_from.split(':'))), dt.time(*map(int, time_at_tt_to.split(':')))
            if time_at_tt_from <= time_time <= time_at_tt_to:
                continue
            else:
                before_minutes = (time_at_tt_from.hour - time_time.hour) * 60 + (time_at_tt_from.minute - time_time.minute)
                after_minutes = (time_time.hour - time_at_tt_to.hour) * 60 + (time_time.minute - time_at_tt_to.minute)
                dist_minutes = max(before_minutes, after_minutes)
                if dist_minutes <= 2 * 60:
                    rate += 3
                else:
                    rate += 1
        else:
            rate += 1

    return rate


if __name__ == '__main__':
    group_bank = GroupBank('data/groups.csv')
    user_bank = UserBank('data/users.csv')
    with open('data/attendance.pkl', 'rb') as f:
        attendance = pickle.load(f)
    feature_creator = FeatureCreator(attendance, group_bank, user_bank)
    checkpoint_file = 'data/checkpoint.pt'
    print(f'LOADING CHECKPOINT {checkpoint_file}')
    state = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    ml_model = AllVectorizer(group_bank, user_bank).eval()
    print(ml_model.load_state_dict(state['model']))
    col = AttendanceCollator()
    del state
    infer_users = pd.read_csv('data/infer_users.csv', dtype={'уникальный номер участника': str})
    infer_users = infer_users.drop(columns=['уникальный номер группы'])
    infer_groups = set(pd.read_csv('data/infer_ids.csv', header=None, dtype={0: str})[0])
    user_recommendations = {}
    for user_id in tqdm(infer_users['уникальный номер участника']):
        features = feature_creator.get_features(user_id, return_target=False, target_stop_random=False)
        features = col([features])
        times = feature_creator.get_attend_times(user_id)
        this_user_microg = []
        with torch.inference_mode():
            macrog = ml_model(features)[0]
            macrog = torch.sort(macrog, descending=True)
            for score, group_id in zip(macrog.values, macrog.indices):
                score = score.item()
                group_id = group_id.item()
                group_microgroups = group_bank.get_group(group_id)
                recom_groups = group_microgroups.intersection(infer_groups)
                recom_timetables = {x: group_bank.get_timetable(x, None) for x in recom_groups}
                recom_match_rate = {x: _get_match_rate(times, recom_timetables[x]) for x in recom_groups}
                recom_groups_ranked = []
                for i, (group_name, group_rate) in enumerate(sorted(recom_match_rate.items(), key=lambda x: x[1], reverse=True)):
                    if i == 3:
                        break
                    recom_groups_ranked.append(group_name)
                this_user_microg.extend(recom_groups_ranked)
        user_recommendations[user_id] = this_user_microg
    user_recommendations = pd.Series({k: ','.join(v[:10]) for k, v in user_recommendations.items()})
    user_recommendations.name = 'уникальный номер группы'
    infer_users = infer_users.join(user_recommendations, on='уникальный номер участника')
    infer_users.to_csv('data/infer_users_out.csv', index=False)
