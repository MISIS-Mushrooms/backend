import pickle

import pandas as pd
import torch

from dedformer.model import GroupBank, UserBank, AllVectorizer, AttendanceCollator, FeatureCreator

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
    infer_groups = set(pd.read_csv('data/infer_ids.csv', header=None, dtype={0: str})[0])
    user_recommendations = {}
    for user_id in infer_users['уникальный номер участника']:
        features = feature_creator.get_features(user_id, return_target=False, target_stop_random=False)
        features = col([features])
        this_user_microg = []
        with torch.inference_mode():
            macrog = ml_model(features)[0]
            macrog = torch.sort(macrog, descending=True)
            for score, group_id in zip(macrog.values, macrog.indices):
                score = score.item()
                group_id = group_id.item()
                group_microgroups = group_bank.get_group(group_id)
                recom_groups = group_microgroups.intersection(infer_groups)
                print(f'got {len(recom_groups)} groups, score={score}')
                this_user_microg.extend(recom_groups)
        user_recommendations[user_id] = this_user_microg
    print(123)
