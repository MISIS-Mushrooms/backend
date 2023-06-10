import pickle
import random
import traceback
from collections import defaultdict
from typing import Optional, List, Literal, Set, Dict

import geopy.distance
import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
import datetime as dt

from tqdm import tqdm

from dedformer.model import UserBank, GroupBank, AllVectorizer, FeatureCreator, AttendanceCollator


class RecommendTags(BaseModel):
    category: Literal['mind'] | Literal['soul'] | Literal['body']
    small_groups: bool = Field(alias='smallGroups')
    next_house: bool = Field(alias='nextHouse')
    new: bool
    online: bool

    class Config:
        allow_population_by_field_name = True


class IdentifyInputs(BaseModel):
    first_name: Optional[str] = Field(None, alias='firstName')
    middle_name: Optional[str] = Field(None, alias='middleName')
    last_name: Optional[str] = Field(None, alias='lastName')
    date_of_birth: Optional[dt.date] = Field(None, alias='dateOfBirth')
    user_id: Optional[str] = Field(None, alias='userId')


class HistoryVariant(BaseModel):
    id: str
    area: List[str]
    visited_at: dt.datetime = Field(alias='visitedAt')

    class Config:
        allow_population_by_field_name = True


class HistoryItem(BaseModel):
    id: str
    category1: str
    category2: str
    category3: str
    description: str
    variant: HistoryVariant



class IdentifyOutputs(BaseModel):
    first_name: Optional[str] = Field(None, alias='firstName')
    middle_name: Optional[str] = Field(None, alias='middleName')
    last_name: Optional[str] = Field(None, alias='lastName')
    date_of_birth: Optional[dt.date] = Field(None, alias='dateOfBirth')
    need_onboarding: bool = Field(alias='needOnboarding')
    user_id: str = Field(alias='userId')
    areas: List[str] = Field(alias='areas')
    history: List[HistoryItem]

    class Config:
        allow_population_by_field_name = True


class RecommendCategories(BaseModel):
    soul: bool
    mind: bool
    body: bool


class RecommendFilterDays(BaseModel):
    mon: bool = True
    tue: bool = True
    wed: bool = True
    thu: bool = True
    fri: bool = True
    sat: bool = True
    sun: bool = True


class RecommendFilters(BaseModel):
    query: str = Field()
    health_problems: bool = Field(alias='healthProblems')
    prefer_online: bool = Field(alias='preferOnline')
    with_grandson: bool = Field(alias='withGrandson')
    categories: RecommendCategories
    areas: Set[str]
    friend_ids: Set[str] = Field(alias='friendIds')
    days: RecommendFilterDays = Field(default_factory=RecommendFilterDays)


class RecommendOnboarding(BaseModel):
    categories: List[str]


class RecommendInputs(BaseModel):
    user_id: str = Field(alias='userId')
    filters: RecommendFilters
    onboarding: Optional[RecommendOnboarding] = Field(None)
    return_variants: bool = Field(True)
    ratings: Dict[str, Literal['liked'] | Literal['disliked']] = Field(default_factory=dict)


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
    distance: float
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


class SomebodyInputs(BaseModel):
    pass


class SomebodyOutputs(BaseModel):
    first_name: str = Field(alias='firstName')
    middle_name: Optional[str] = Field(alias='middleName')
    last_name: str = Field(alias='lastName')
    date_of_birth: dt.date = Field(alias='dateOfBirth')

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


def load_full_attendance():
    attendance = pd.read_csv('data/attend.csv', dtype={
        'уникальный номер занятия': str,
        'уникальный номер группы': str,
        'уникальный номер участника': str,
        'дата занятия': str,
        'время начала занятия': str,
        'время окончания занятия': str
    })
    tqdm.pandas()
    attendance['dt'] = attendance.progress_apply(
        lambda x: dt.datetime.fromisoformat(f'{x["дата занятия"]}T{x["время начала занятия"]}'), axis=1)
    attendance = attendance[['уникальный номер группы', 'уникальный номер участника', 'dt']]
    attendance = attendance.groupby('уникальный номер участника').apply(lambda x: x.drop(columns=['уникальный номер участника']).to_dict(orient='records')).to_dict()
    return attendance


def create_app():
    history = load_full_attendance()
    user_bank = UserBank('data/users.csv', 'data/user_names.csv')
    group_bank = GroupBank('data/groups.csv')
    with open('data/attendance.pkl', 'rb') as f:
        attendance = pickle.load(f)
    feature_creator = FeatureCreator(attendance, group_bank, user_bank)
    app = FastAPI()
    checkpoint_file = 'data/checkpoint.pt'
    print(f'LOADING CHECKPOINT {checkpoint_file}')
    state = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    ml_model = AllVectorizer(group_bank, user_bank).eval()
    print(ml_model.load_state_dict(state['model']))
    col = AttendanceCollator()
    del state

    @app.post('/identify')
    async def identify(inputs: IdentifyInputs) -> IdentifyOutputs:
        if inputs.user_id is None:
            uid = user_bank.get_user_id_by_name(inputs.date_of_birth, inputs.first_name, inputs.middle_name, inputs.last_name)
            if uid is not None:
                first, middle, last, dow = inputs.first_name, inputs.middle_name, inputs.last_name, inputs.date_of_birth
            else:
                first = middle = last = ''
                dow = dt.date(1970, 1, 1)
        else:
            uid = inputs.user_id
            ident = user_bank.get_user_identity_by_id(uid)
            first, middle, last, dow = ident['first'], ident['middle'], ident['last'], ident['дата рождения']
        user_hist = history.get(uid, None)
        hist = []
        if user_hist is not None:
            for itm in user_hist:
                try:
                    group_id = itm['уникальный номер группы']
                    timestamp = itm['dt'].to_pydatetime()
                    macro_id = group_bank.get_macro_group_id(group_id)
                    group_blocks = all_blocks[macro_id]
                    group_actions = all_actions[0][macro_id], all_actions[1][macro_id], all_actions[2][macro_id]
                    group_description = all_descriptions[macro_id]
                    hist.append(HistoryItem(
                        id=str(hash(group_actions)),
                        category1=group_actions[0],
                        category2=group_actions[1],
                        category3=group_actions[2],
                        description=group_description,
                        variant=HistoryVariant(id=str(group_id), area=list(group_blocks), visited_at=timestamp)
                    ))
                except:
                    traceback.print_exc()
        hist = sorted(hist, key=lambda x: x.variant.visited_at)
        return IdentifyOutputs(
            areas=group_bank.get_all_blocks(),
            user_id='demo' if uid is None else uid,
            need_onboarding=uid is None,
            history=hist,
            first_name=first,
            middle_name=middle,
            last_name=last,
            date_of_birth=dow
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
    all_search_str = [' '.join(x.strip() for x in zippd).lower() for zippd in zip(*all_actions, all_descriptions)]
    today_timetables = {k: group_bank.get_timetable(k, dt.date(2023, 4, 10)) for k in group_bank.get_all_group_ids()}

    @app.post('/recommend')
    async def recommend(inputs: RecommendInputs) -> RecommendOutputs:
        user_geo = user_bank.get_geo(inputs.user_id)
        user_demo_mode = user_geo is None
        if user_geo is None:
            user_geo = (55.7385557, 37.6183149)  # третьяковка в центре москвы
        if inputs.user_id in user_bank.get_all_user_ids():
            features = col([feature_creator.get_features(x, return_target=False, target_stop_random=False) for x in [inputs.user_id, *inputs.filters.friend_ids]])
            with torch.inference_mode():
                ml_similarity = (ml_model(features, mean_vec_along_batch=True)[0] + 1) / 2
            ml_similarity = ml_similarity.clone()
        else:
            user_rng = random.Random(x=inputs.user_id)
            ml_similarity = torch.FloatTensor([user_rng.uniform(0.48, 0.54) for _ in range(group_bank.num_macro_groups)])

        for cat in inputs.onboarding.categories:
            ml_similarity[torch.BoolTensor([x == cat for x in all_actions[0]])] *= 1.5

        q = inputs.filters.query.strip().lower()
        if q != '':
            search_mask = torch.BoolTensor([q in x for x in all_search_str])
            ml_similarity[~search_mask] = 0
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
            groups_with_timetables = {k: RecommendTimetable.parse_obj(today_timetables[k]) for k in group_bank.get_group(macro_group_index) if today_timetables[k] is not None}
            groups_distances = {k: (geopy.distance.distance(user_geo, group_bank.get_geocoord(k)).meters if (user_geo is not None and not group_online) else 0) for k in groups_with_timetables.keys()}
            variants = [RecommendVariant(id=idx, timetable=tt, area=list(group_blocks), distance=groups_distances[idx]) for idx, tt in groups_with_timetables.items()]

            def filter_variant(variant: RecommendVariant, days: RecommendFilterDays):
                day_pairs = [
                    (days.mon, variant.timetable.mon),
                    (days.tue, variant.timetable.tue),
                    (days.wed, variant.timetable.wed),
                    (days.thu, variant.timetable.thu),
                    (days.fri, variant.timetable.fri),
                    (days.sat, variant.timetable.sat),
                    (days.sun, variant.timetable.sun),
                ]
                for pair in day_pairs:
                    if pair[0] and (pair[1] != 'нет'):  # если чел может и в расписании есть день, то ок
                        return True
                return False

            variants = [x for x in variants if filter_variant(x, inputs.filters.days)]
            variants = sorted(variants, key=lambda x: (x.distance, x.timetable.mon, x.timetable.tue, x.timetable.wed, x.timetable.thu, x.timetable.fri, x.timetable.sat, x.timetable.sun))
            if len(variants) == 0:
                continue

            if group_actions not in mega_items:
                mega_items[group_actions] = {
                    'id': str(hash(group_actions)),
                    'category1': group_actions[0],
                    'category2': group_actions[1],
                    'category3': group_actions[2],
                    'description': group_description,
                    'tags': RecommendTags(category=group_macrocat, small_groups=False, next_house=False, new=False,
                                          online=group_online)
                }
            if not group_online:
                min_distance = min(map(lambda x: x.distance, variants))
                if min_distance <= 350:
                    mega_items[group_actions]['tags'].next_house = True
            mega_group_variants[group_actions].extend(variants)

        if user_demo_mode:
            for k in mega_group_variants:
                mega_group_variants[k] = sorted(mega_group_variants[k], key=lambda x: (x.distance, x.timetable.mon, x.timetable.tue, x.timetable.wed, x.timetable.thu, x.timetable.fri,x.timetable.sat, x.timetable.sun))



        return RecommendOutputs(
            items=[RecommendItem(variants=variants if inputs.return_variants else [], **mega_items[k]) for k, variants in mega_group_variants.items()][:50]
        )

    @app.post('/somebody')
    async def somebody(inputs: SomebodyInputs):
        sb = user_bank.get_somebody()
        return SomebodyOutputs(first_name=sb['first'], middle_name=sb['middle'], last_name=sb['last'], date_of_birth=sb['birth'])

    return app
