import random
import re
import warnings
from collections import defaultdict
from typing import Tuple, Set, Dict, List, Optional, FrozenSet

import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning.distances import CosineSimilarity
from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import torch.nn.functional as F
import datetime as dt

from torchmetrics import Metric, MeanMetric, Accuracy
from torchmetrics.functional import pairwise_cosine_similarity
from tqdm import tqdm
from transformers import RobertaConfig, T5Config
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaPooler, RobertaSelfAttention
from xztrainer import XZTrainable, ContextType, BaseContext, DataType, ModelOutputsType

from dedformer.arcface import ArcFace
from dedformer.t5 import T5ForConditionalGeneration

_RE_ADM_AREA = re.compile(r'административн\w+ округ\w*')
_RE_WORD = re.compile(r'[\w-]+')


class UserBank:
    def __init__(self, users_path):
        df = pd.read_csv(users_path, dtype={
            'уникальный номер': str,
            'дата создания личного дела': str,
            'пол': str,
            'дата рождения': str,
            'адрес проживания': str
        })
        df['дата создание личного дела'] = df['дата создание личного дела'].apply(lambda x: dt.datetime.fromisoformat(x.strip().replace(' ', 'T')))
        df['дата рождения'] = df['дата рождения'].apply(lambda x: dt.date.fromisoformat(x))
        df = df.set_index('уникальный номер')
        self._df = df

    @property
    def n_age_bins(self):
        return 7

    @property
    def n_genders(self):
        return 2

    def to_model_data(self, idx: str):
        sample = self._df.loc[idx]
        age_years = ((sample['дата создание личного дела'].date() - sample['дата рождения']).days // 365)
        age_bin = min(16, max(10, age_years // 5)) - 10
        return {
            'user_gender': torch.scalar_tensor(sample['пол'] == 'Женщина', dtype=torch.long),
            'user_age_bin': torch.scalar_tensor(age_bin, dtype=torch.long),

        }



class GroupBank:
    def __init__(self, groups_path):
        df = pd.read_csv(groups_path, dtype={
            'уникальный номер': str,
            'направление 1': str,
            'направление 2': str,
            'направление 3': str,
            'адрес площадки': str,
            'округ площадки': str,
            'район площадки': str  # todo вектор расписания
        }).set_index('уникальный номер')
        df['округ площадки'] = df['округ площадки'].apply(self._cleanup_area)
        df['район площадки'] = df['район площадки'].apply(self._cleanup_block)
        df['is_online'] = df['направление 3'].str.lower().str.contains('онлайн')
        self._df = df
        self._all_actions = set(self._df['направление 1'].unique()), set(self._df['направление 2'].unique()), set(self._df['направление 3'].unique())
        self._all_areas = set(self._df['округ площадки'].explode().unique())
        self._all_blocks = set(self._df['район площадки'].explode().unique())
        self._action_id = self._generate_id_map(self._all_actions[0]), self._generate_id_map(self._all_actions[1]), self._generate_id_map(self._all_actions[2])
        self._area_id = self._generate_id_map(self._all_areas)
        self._block_id = self._generate_id_map(self._all_blocks)
        group_by_cols = ['направление 1', 'направление 2', 'направление 3', 'округ площадки', 'район площадки']
        self._macro_groups = {i: {'ids': set(x[1].index), 'features': {k: v for k, v in zip(group_by_cols, x[0])}} for i, x in enumerate(df.groupby(group_by_cols))}
        self._group_to_macro_group = {idx: index for index, itms in self._macro_groups.items() for idx in itms['ids']}
        print(123)

    @staticmethod
    def _generate_id_map(s: Set[str]) -> Dict[str, int]:
        return {s: i for i, s in enumerate(s)}

    @staticmethod
    def _cleanup_area(s: str) -> FrozenSet[str]:
        if pd.isna(s) or s.strip() == '':
            return frozenset()
        s = _RE_ADM_AREA.sub(' ', s)
        s = s.replace(' и ', ' ')
        s = _RE_WORD.findall(s)
        s = frozenset(s)
        return s

    @staticmethod
    def _cleanup_block(s: str) -> FrozenSet[str]:
        if pd.isna(s) or s.strip() == '':
            return frozenset()
        return frozenset(s.split(', '))

    def get_action_by_ids(self, macro_group_id: List[int]) -> Tuple[List[str], List[str], List[str]]:
        lst = [[self._macro_groups[mid]['features']['направление 1'], self._macro_groups[mid]['features']['направление 2'], self._macro_groups[mid]['features']['направление 3']] for mid in macro_group_id]
        return [x[0] for x in lst], [x[1] for x in lst], [x[2] for x in lst]

    def get_area_by_ids(self, macro_group_id: List[int]) -> List[Set[str]]:
        return [self._macro_groups[mid]['features']['округ площадки'] for mid in macro_group_id]

    def get_block_by_ids(self, macro_group_id: List[int]) -> List[Set[str]]:
        return [self._macro_groups[mid]['features']['район площадки'] for mid in macro_group_id]

    def get_all_actions(self) -> Tuple[Set[str], Set[str], Set[str]]:
        return self._all_actions

    def get_all_areas(self) -> Set[str]:
        return self._all_areas

    def macro_group_num(self) -> int:
        return len(self._macro_groups)

    def get_all_blocks(self) -> Set[str]:
        return self._all_blocks

    def get_macro_group_id(self, idx: str) -> int:
        return self._group_to_macro_group[idx]

    def get_model_data(self, macro_group_ids: List[int]):
        actions = self.get_action_by_ids(macro_group_ids)
        areas = self.get_area_by_ids(macro_group_ids)
        areas = [list(x) for x in areas]
        blocks = self.get_block_by_ids(macro_group_ids)
        blocks = [list(x) for x in blocks]
        max_area_len = max(map(len, areas))
        max_block_len = max(map(len, blocks))
        area_ids = [[self._area_id[a[i]] if i < len(a) else 0 for i in range(max_area_len)] for a in areas]
        area_mask = [[1 if i < len(a) else 0 for i in range(max_area_len)] for a in areas]
        block_ids = [[self._block_id[b[i]] if i < len(b) else 0 for i in range(max_block_len)] for b in blocks]
        block_mask = [[1 if i < len(b) else 0 for i in range(max_block_len)] for b in blocks]
        return {
            'action_1': torch.LongTensor([self._action_id[0][x] for x in actions[0]]),
            'action_2': torch.LongTensor([self._action_id[1][x] for x in actions[1]]),
            'action_3': torch.LongTensor([self._action_id[2][x] for x in actions[2]]),
            'area': torch.LongTensor(area_ids),
            'area_mask': torch.LongTensor(area_mask),
            'block': torch.LongTensor(block_ids),
            'block_mask': torch.LongTensor(block_mask)
        }

    @property
    def num_macro_groups(self):
        return len(self._macro_groups)


class AttendanceDataset(Dataset):
    def __init__(self, attend: Dict, bank_group: GroupBank, bank_user: UserBank, is_train: bool, dummy: bool):
        self._bank_group = bank_group
        self._bank_user = bank_user
        self._attend = attend
        self._attend_indices = [k for k in attend.keys()]
        self._is_train = is_train
        self._dummy = dummy

    def __getitem__(self, index):
        user_id = self._attend_indices[index]
        user_data = self._bank_user.to_model_data(user_id)
        item = self._attend[user_id]

        visited = set()
        group_nums = []
        target_first_seen = None
        target = None
        for row in item.itertuples():
            if row.macro_id not in visited:
                visited.add(row.macro_id)
                group_nums.append(row.macro_id)
                target_first_seen = row.dt
                target = row.macro_id
        group_nums = group_nums[:-1]

        group_data_by_macro = item.groupby('macro_id')
        group_times, group_min_dt, group_max_dt, group_often_time = [], [], [], []
        for mac in group_nums:
            mac_data = group_data_by_macro.get_group(mac)
            mac_data = mac_data[mac_data['dt'] <= target_first_seen]
            group_times.append(int(mac_data['dt'].count()))
            group_min_dt.append(mac_data['dt'].min().to_pydatetime())
            group_max_dt.append(mac_data['dt'].max().to_pydatetime())
            # group_often_time.append(mac_data['dt'].dt.time.mode().iloc[0])
        group_times_sum = sum(group_times)
        if group_times_sum == 0 and len(group_times) > 0:
            print(123)


        # if self._is_train:
        #     min_length = 1
        #     group_start = random.randint(0, len(group_nums) - min_length)
        #     group_end = group_start + random.randint(group_start, len(group_nums) - 1)
        #     group_nums = group_nums[group_start:group_end + 1]
        return {
            'group_sequence': torch.LongTensor(group_nums),
            'group_popularity': torch.FloatTensor([x / group_times_sum for x in group_times]),
            'group_sequence_out': torch.LongTensor(group_nums + [target]),
            'group_target': torch.scalar_tensor(target, dtype=torch.long),
            'group_pos': torch.LongTensor(list(range(len(group_nums) + 1))[::-1]),
            'group_mask': torch.LongTensor([1 for _ in range(len(group_nums) + 1)]),
            **user_data
        }

    def __len__(self):
        return len(self._attend)


class AttendanceCollator():
    def __call__(self, batch):
        max_group_mask_ln = max([len(x['group_mask']) for x in batch])
        max_group_sequence_ln = max([len(x['group_sequence']) for x in batch])
        max_group_sequence_ln_out = max([len(x['group_sequence_out']) for x in batch])
        return {
            'group_sequence': torch.stack([F.pad(x['group_sequence'], (0, max_group_sequence_ln - len(x['group_sequence']))) for x in batch]),
            'group_target': torch.stack([x['group_target'] for x in batch]),
            'group_pos': torch.stack([F.pad(x['group_pos'], (0, max_group_sequence_ln - len(x['group_pos']))) for x in batch]),
            'group_mask': torch.stack([F.pad(x['group_mask'], (0, max_group_mask_ln - len(x['group_mask']))) for x in batch]),
            'group_sequence_out': torch.stack([F.pad(x['group_sequence_out'], (0, max_group_sequence_ln_out - len(x['group_sequence_out']))) for x in batch]),
            'user_gender': torch.stack([x['user_gender'] for x in batch]),
            'user_age_bin': torch.stack([x['user_age_bin'] for x in batch]),
            'group_popularity': torch.stack([F.pad(x['group_popularity'], (0, max_group_sequence_ln - len(x['group_popularity']))) for x in batch])
        }


EMBED_DIM = 256


class UserVectorizer(nn.Module):
    def __init__(self, bank: UserBank):
        super().__init__()

        self._cls = nn.Parameter(torch.empty([1, EMBED_DIM]))
        nn.init.normal_(self._cls)

        self._gender_emb = nn.Embedding(bank.n_genders, EMBED_DIM)
        self._age_emb = nn.Embedding(bank.n_age_bins, EMBED_DIM)

    def forward(self, data):
        gender_emb = self._gender_emb(data['user_gender'])
        age_emb = self._age_emb(data['user_age_bin'])
        cls_emb = self._cls.repeat(gender_emb.shape[0], 1)
        all_emb = torch.stack([cls_emb, gender_emb, age_emb], dim=1)
        return all_emb, torch.ones([all_emb.shape[0], all_emb.shape[1]], device=all_emb.device, dtype=torch.long)


class GroupVectorizer(nn.Module):
    def __init__(self, bank: GroupBank):
        super().__init__()
        all_actions = bank.get_all_actions()
        all_areas = bank.get_all_areas()
        all_blocks = bank.get_all_blocks()
        self._act_1_emb = nn.Embedding(len(all_actions[0]), EMBED_DIM)
        self._act_2_emb = nn.Embedding(len(all_actions[1]), EMBED_DIM)
        self._act_3_emb = nn.Embedding(len(all_actions[2]), EMBED_DIM)
        self._area_emb = nn.Embedding(len(all_areas), EMBED_DIM)
        self._block_emb = nn.Embedding(len(all_blocks), EMBED_DIM)
        self._aten = RobertaSelfAttention(RobertaConfig(hidden_size=EMBED_DIM, num_attention_heads=2))

    def forward(self, data):
        act_1_emb = self._act_1_emb(data['action_1'])
        act_2_emb = self._act_2_emb(data['action_2'])
        act_3_emb = self._act_3_emb(data['action_3'])
        area_emb = self._area_emb(data['area'])
        block_emb = self._block_emb(data['block'])
        area_emb = (data['area_mask'].unsqueeze(2) * area_emb).sum(dim=1) / data['area_mask'].sum(dim=1).clamp(min=1).unsqueeze(1)
        block_emb = (data['block_mask'].unsqueeze(2) * block_emb).sum(dim=1) / data['block_mask'].sum(dim=1).clamp(min=1).unsqueeze(1)
        all_emb = torch.stack([act_1_emb, act_2_emb, act_3_emb, area_emb, block_emb], dim=1)
        all_emb = self._aten(all_emb)[0].sum(dim=1)
        return all_emb


def get_extended_attention_mask(
    attention_mask: torch.Tensor, dtype: torch.float = None
) -> torch.Tensor:
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


class AllVectorizer(nn.Module):
    def __init__(self, bank: GroupBank, bank_users: UserBank):
        super().__init__()
        self._group_vec = GroupVectorizer(bank)
        self._user_vec = UserVectorizer(bank_users)
        self._all_group_index = nn.Parameter(torch.LongTensor(list(range(bank.num_macro_groups))), requires_grad=False)
        self._all_group_data = nn.ParameterDict({k: nn.Parameter(v, requires_grad=False) for k, v in bank.get_model_data(list(range(bank.num_macro_groups))).items()})
        self._popularity_enc = nn.Sequential(nn.Linear(1, EMBED_DIM), nn.GELU(), nn.Linear(EMBED_DIM, EMBED_DIM))
        self._sos = nn.Parameter(torch.empty([1, 1, EMBED_DIM]))
        nn.init.normal_(self._sos)
        self._pos_enc = nn.Embedding(512, EMBED_DIM)
        cfg = T5Config(
            d_model=EMBED_DIM,
            d_kv=EMBED_DIM // 4,
            num_heads=4,
            d_ff=EMBED_DIM * 4,
            num_layers=1,
            num_decoder_layers=4,
            intermediate_size=EMBED_DIM * 4,
            tie_word_embeddings=False
        )
        self._t5 = T5ForConditionalGeneration(cfg)
        self._loss = ArcFace()
        self._decoder_feature_aten = RobertaSelfAttention(RobertaConfig(hidden_size=EMBED_DIM, num_attention_heads=1))
        # self._loss = TripletMarginLoss(distance=CosineSimilarity())

    def forward(self, data):
        group_seq = data['group_sequence']
        group_pos = data['group_pos']
        group_mask = data['group_mask']
        group_target = data['group_target']
        group_seq_out = data['group_sequence_out']
        all_group_vec = self._group_vec(self._all_group_data)
        x_dec = all_group_vec[group_seq]
        # pos_ids = torch.arange(curr_vec.shape[1] - 1, -1, -1)
        x_dec = x_dec + self._pos_enc(group_pos)
        # x_dec = self._decoder_feature_aten(
        #     torch.stack([
        #         x_dec,
        #         self._pos_enc(group_pos),
        #         self._popularity_enc(data['group_popularity'].unsqueeze(2))
        #     ], dim=-2).view(-1, 3, EMBED_DIM))[0].sum(dim=1).view(group_seq.shape[0], group_seq.shape[1], EMBED_DIM)
        x_dec = torch.cat([self._sos.repeat(x_dec.shape[0], 1, 1), x_dec], dim=1)
        enc_x, enc_mask = self._user_vec(data)
        # group_mask = get_extended_attention_mask(group_mask, dtype=x.dtype)
        x = self._t5(input_embeds=enc_x, input_attention_mask=enc_mask, decoder_inputs_embeds=x_dec,
                     decoder_attention_mask=group_mask).logits
        loss, logits = self._loss(all_group_vec, x[group_mask == 1], group_seq_out[group_mask == 1])
        return loss, logits[group_mask.sum(dim=1).cumsum(0) - 1]


class MyTrainable(XZTrainable):
    def __init__(self, bank: GroupBank):
        self._bank = bank

    def step(self, context: BaseContext, data: DataType) -> Tuple[Tensor, ModelOutputsType]:
        loss, logits = context.model(data)
        return loss, {
            'loss': loss,
            'target': data['group_target'],
            'proba': torch.softmax(logits, dim=1)
        }

    def create_metrics(self, context_type: ContextType) -> Dict[str, Metric]:
        return {
            'loss': MeanMetric(),
            'accuracy_top1': Accuracy('multiclass', num_classes=self._bank.macro_group_num()),
            'accuracy_top5': Accuracy('multiclass', num_classes=self._bank.macro_group_num(), top_k=5),
            'accuracy_top10': Accuracy('multiclass', num_classes=self._bank.macro_group_num(), top_k=10)
        }

    def update_metrics(self, context_type: ContextType, model_outputs: Dict[str, List], metrics: Dict[str, Metric]):
        metrics['loss'].update(model_outputs['loss'])
        metrics['accuracy_top1'].update(model_outputs['proba'], model_outputs['target'])
        metrics['accuracy_top5'].update(model_outputs['proba'], model_outputs['target'])
        metrics['accuracy_top10'].update(model_outputs['proba'], model_outputs['target'])
