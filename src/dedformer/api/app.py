import pickle
import random
from collections import defaultdict
from typing import Optional, List, Literal, Set

import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
import datetime as dt

from dedformer.model import UserBank, GroupBank, AllVectorizer, FeatureCreator, AttendanceCollator


class IdentifyInputs(BaseModel):
    first_name: str = Field(alias='firstName')
    middle_name: Optional[str] = Field(alias='middleName')
    last_name: str = Field(alias='lastName')
    date_of_birth: dt.date = Field(alias='dateOfBirth')


class IdentifyOutputs(BaseModel):
    need_onboarding: bool = Field(alias='needOnboarding')
    user_id: str = Field(alias='userId')
    areas: List[str] = Field(alias='areas')

    class Config:
        allow_population_by_field_name = True


class RecommendCategories(BaseModel):
    soul: bool
    mind: bool
    body: bool


class RecommendFilters(BaseModel):
    health_problems: bool = Field(alias='healthProblems')
    prefer_online: bool = Field(alias='preferOnline')
    with_grandson: bool = Field(alias='withGrandson')
    categories: RecommendCategories
    areas: Set[str]


class RecommendOnboarding(BaseModel):
    categories: List[str]


class RecommendInputs(BaseModel):
    user_id: str = Field(alias='userId')
    filters: RecommendFilters
    onboarding: Optional[RecommendOnboarding] = Field(None)
    return_variants: bool = Field(True)


class RecommendTags(BaseModel):
    category: Literal['mind'] | Literal['soul'] | Literal['body']
    small_groups: bool = Field(alias='smallGroups')
    next_house: bool = Field(alias='nextHouse')
    new: bool
    online: bool

    class Config:
        allow_population_by_field_name = True


class RecommendTimetable(BaseModel):
    mon: str
    tue: str
    wed: str
    thu: str
    fri: str
    sat: str
    sun: str


class RecommendVariant(BaseModel):
    id: str
    area: List[str]
    timetable: RecommendTimetable



class RecommendItem(BaseModel):
    id: str
    category1: str
    category2: str
    category3: str
    description: str
    tags: RecommendTags
    variants: List[RecommendVariant]


class RecommendOutputs(BaseModel):
    items: List[RecommendItem]
    class Config:
        allow_population_by_field_name = True


DROP_HEALTH_PROBLEMS = {'Физическая активность'}
GRANDSON_CATEGORIES = {'ОНЛАЙН Рисование', 'ОНЛАЙН Английский язык', 'Иностранные языки', 'ОНЛАЙН Интеллектуальный клуб. Информационные технологии',
                       'Интеллектуальный клуб. Иностранные языки', 'ОНЛАЙН Пение', 'Настольные игры',
                       'Интеллектуальный клуб. Информационные технологии', 'ОНЛАЙН Шахматы и шашки',
                       'Интеллектуальные игры', 'ОНЛАЙН Интеллектуальный клуб. Иностранные языки',
                       'Шахматы и шашки', 'ОНЛАЙН Информационные технологии',
                       'Фото/видео', 'ОНЛАЙН Пеший лекторий', 'Пение',
                       'ОНЛАЙН Настольные игры', 'Пеший лекторий',
                       'ОНЛАЙН Интеллектуальные игры', 'ОНЛАЙН Московский театрал',
                       'ОНЛАЙН Фото/видео', 'Лыжи',
                       'Московский театрал', 'Киберспорт',
                       'Коньки', 'ОНЛАЙН Интеллектуальный клуб. Творческие мастерские',
                       'ОНЛАЙН Киберспорт', 'Информационные технологии',
                       'Интеллектуальный клуб. Творческие мастерские', 'Английский язык',
                       'Рисование', 'ОНЛАЙН Иностранные языки',
                       'Художественно-прикладное творчество'}


def create_app():
    user_bank = UserBank('data/users.csv', 'data/user_names.csv')
    group_bank = GroupBank('data/groups.csv')
    with open('data/attendance.pkl', 'rb') as f:
        attendance = pickle.load(f)
    feature_creator = FeatureCreator(attendance, group_bank, user_bank)
    app = FastAPI()
    checkpoint_file = 'checkpoint/t5-users-onlinevec-posfix-last-aug-lizafts-fix2/save-14320.pt'
    print(f'LOADING CHECKPOINT {checkpoint_file}')
    state = torch.load(checkpoint_file)
    ml_model = AllVectorizer(group_bank, user_bank).eval()
    print(ml_model.load_state_dict(state['model']))
    col = AttendanceCollator()
    del state

    @app.post('/identify')
    async def identify(inputs: IdentifyInputs) -> IdentifyOutputs:
        uid = user_bank.get_user_id_by_name(inputs.date_of_birth, inputs.first_name, inputs.middle_name, inputs.last_name)
        return IdentifyOutputs(
            areas=group_bank.get_all_blocks(),
            user_id='demo' if uid is None else uid,
            need_onboarding=uid is None
        )

    all_actions = group_bank.get_action_by_ids(list(range(group_bank.num_macro_groups)))
    all_block_set = group_bank.get_all_blocks()
    all_blocks = group_bank.get_block_by_ids(list(range(group_bank.num_macro_groups)))
    all_threecatts = group_bank.get_threecat_by_ids(list(range(group_bank.num_macro_groups)))
    health_mask = torch.BoolTensor([x in DROP_HEALTH_PROBLEMS for x in all_actions[0]])
    all_onlines = group_bank.get_action_online_by_ids(list(range(group_bank.num_macro_groups)))
    all_descriptions = group_bank.get_description_by_ids(list(range(group_bank.num_macro_groups)))
    online_mask = torch.BoolTensor([x for x in all_onlines])
    grandson_mask = torch.BoolTensor([x in GRANDSON_CATEGORIES for x in all_actions[1]])
    threecat_mind_mask = torch.BoolTensor([x == 'mind' for x in all_threecatts])
    threecat_soul_mask = torch.BoolTensor([x == 'soul' for x in all_threecatts])
    threecat_body_mask = torch.BoolTensor([x == 'body' for x in all_threecatts])
    threecat_area_mask = {block: torch.BoolTensor([
        block in x for x in all_blocks
    ]) for block in all_block_set}
    today_timetables = {k: group_bank.get_timetable(k, dt.date(2023, 4, 10)) for k in group_bank.get_all_group_ids()}
    print(123)

    @app.post('/recommend')
    async def recommend(inputs: RecommendInputs) -> RecommendOutputs:
        if inputs.user_id in user_bank.get_all_user_ids():
            features = col([feature_creator.get_features(inputs.user_id, return_target=False, target_stop_random=False)])
            with torch.inference_mode():
                ml_similarity = (ml_model(features)[0] + 1) / 2
            ml_similarity = ml_similarity.clone()
        else:
            ml_similarity = torch.FloatTensor([random.uniform(0.48, 0.54) for _ in range(group_bank.num_macro_groups)])

        for cat in inputs.onboarding.categories:
            ml_similarity[torch.BoolTensor([x == cat for x in all_actions[0]])] *= 1.5

        # trim online


        if inputs.filters.health_problems:
            ml_similarity[health_mask] = 0
        if inputs.filters.prefer_online:
            ml_similarity[online_mask] = ml_similarity[online_mask] * 1.1
        else:
            ml_similarity[online_mask] = ml_similarity[online_mask] * 0.8
        if inputs.filters.with_grandson:
            ml_similarity[grandson_mask] = ml_similarity[grandson_mask] * 1.1
        if not inputs.filters.categories.mind:
            ml_similarity[threecat_mind_mask] = 0
        if not inputs.filters.categories.body:
            ml_similarity[threecat_body_mask] = 0
        if not inputs.filters.categories.soul:
            ml_similarity[threecat_soul_mask] = 0
        if len(inputs.filters.areas) > 0:
            for i, (areas, online) in enumerate(zip(all_blocks, all_onlines)):
                if not online and len(inputs.filters.areas.intersection(areas)) == 0:
                    ml_similarity[i] = 0
        ml_sort = ml_similarity.sort(descending=True)
        mega_items = {}
        mega_group_variants = defaultdict(list)
        for score, macro_group_index in zip(ml_sort.values, ml_sort.indices):
            macro_group_index = macro_group_index.item()
            score = score.item()

            if score == 0:
                break

            group_blocks = all_blocks[macro_group_index]
            group_actions = all_actions[0][macro_group_index], all_actions[1][macro_group_index], all_actions[2][macro_group_index]
            group_macrocat = all_threecatts[macro_group_index]
            group_online = all_onlines[macro_group_index]
            group_description = all_descriptions[macro_group_index]
            group_near = False
            group_small = False
            group_new = False
            groups_with_timetables = {k: RecommendTimetable.parse_obj(today_timetables[k]) for k in group_bank.get_group(macro_group_index) if today_timetables[k] is not None}
            variants = [RecommendVariant(id=idx, timetable=tt, area=list(group_blocks)) for idx, tt in groups_with_timetables.items()]
            variants = sorted(variants, key=lambda x: (x.timetable.mon, x.timetable.tue, x.timetable.wed, x.timetable.thu, x.timetable.fri, x.timetable.sat, x.timetable.sun))
            if group_actions not in mega_items:
                mega_items[group_actions] = {
                    'id': str(hash(group_actions)),
                    'category1': group_actions[0],
                    'category2': group_actions[1],
                    'category3': group_actions[2],
                    'description': group_description,
                    'tags': RecommendTags(category=group_macrocat, small_groups=group_small, next_house=group_near, new=group_new,
                                          online=group_online)
                }
            mega_group_variants[group_actions].extend(variants)


        return RecommendOutputs(
            items=[RecommendItem(variants=variants if inputs.return_variants else [], **mega_items[k]) for k, variants in mega_group_variants.items()][:50]
        )

    return app
