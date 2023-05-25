import re
import warnings
from typing import Tuple, Set, Dict, List, Optional

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
from transformers import RobertaConfig
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaPooler
from xztrainer import XZTrainable, ContextType, BaseContext, DataType, ModelOutputsType

from dedformer.arcface import ArcFace

_RE_ADM_AREA = re.compile(r'административн\w+ округ\w*')
_RE_WORD = re.compile(r'[\w-]+')


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
        self._group_to_num = {k: i for i, k in enumerate(self._df.index)}

    @staticmethod
    def _generate_id_map(s: Set[str]) -> Dict[str, int]:
        return {s: i for i, s in enumerate(s)}

    @staticmethod
    def _cleanup_area(s: str) -> Set[str]:
        if pd.isna(s) or s.strip() == '':
            return set()
        s = _RE_ADM_AREA.sub(' ', s)
        s = s.replace(' и ', ' ')
        s = _RE_WORD.findall(s)
        s = set(s)
        return s

    @staticmethod
    def _cleanup_block(s: str) -> Set[str]:
        if pd.isna(s) or s.strip() == '':
            return set()
        return set(s.split(', '))

    def get_action_by_ids(self, group_id: List[str]) -> Tuple[List[str], List[str], List[str]]:
        lst = list(list(x) for x in self._df.loc[group_id][['направление 1', 'направление 2', 'направление 3']].itertuples(index=False))
        return [x[0] for x in lst], [x[1] for x in lst], [x[2] for x in lst]

    def get_area_by_ids(self, group_id: List[str]) -> List[Set[str]]:
        return list(self._df.loc[group_id]['округ площадки'])

    def get_block_by_ids(self, group_id: List[str]) -> List[Set[str]]:
        return list(self._df.loc[group_id]['район площадки'])

    def get_all_actions(self) -> Tuple[Set[str], Set[str], Set[str]]:
        return self._all_actions

    def get_all_areas(self) -> Set[str]:
        return self._all_areas

    def get_all_ids(self) -> List[str]:
        return list(self._df.index)

    def group_num(self) -> int:
        return len(self._df)

    def get_all_blocks(self) -> Set[str]:
        return self._all_blocks


    def get_model_data(self, group_ids: List[str]):
        actions = self.get_action_by_ids(group_ids)
        areas = self.get_area_by_ids(group_ids)
        areas = [list(x) for x in areas]
        blocks = self.get_block_by_ids(group_ids)
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

    def get_group_numbers(self, groups: List[str]):
        return [self._group_to_num[x] for x in groups]


class AttendanceDataset(Dataset):
    def __init__(self, attend: Dict, bank: GroupBank):
        self._bank = bank
        self._attend = attend
        self._attend_indices = [k for k in attend.keys()]

    def __getitem__(self, index):
        item = self._attend[self._attend_indices[index]]
        group_nums = self._bank.get_group_numbers(item['group_sequence'])
        group_nums = group_nums[::-1][:500][::-1]
        return {
            'group_sequence': torch.LongTensor(group_nums[:-1]),
            'group_target': torch.scalar_tensor(group_nums[-1], dtype=torch.long),
            'group_pos': torch.LongTensor(list(range(len(group_nums)))[::-1]),
            'group_mask': torch.LongTensor([1 for _ in range(len(group_nums))])
        }

    def __len__(self):
        return len(self._attend)


class AttendanceCollator():
    def __call__(self, batch):
        max_group_sequence_ln = max([len(x['group_sequence']) for x in batch])
        return {
            'group_sequence': torch.stack([F.pad(x['group_sequence'], (0, max_group_sequence_ln - len(x['group_sequence']))) for x in batch]),
            'group_target': torch.stack([x['group_target'] for x in batch]),
            'group_pos': torch.stack([F.pad(x['group_pos'], (0, max_group_sequence_ln - len(x['group_pos']))) for x in batch]),
            'group_mask': torch.stack([F.pad(x['group_mask'], (0, max_group_sequence_ln - len(x['group_mask']))) for x in batch])
        }


EMBED_DIM = 256


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
        self._W = nn.Parameter(torch.empty([5, 1, 1]))
        nn.init.uniform_(self._W, 0.3, 0.7)

    def forward(self, data):
        act_1_emb = self._act_1_emb(data['action_1'])
        act_2_emb = self._act_2_emb(data['action_2'])
        act_3_emb = self._act_3_emb(data['action_3'])
        area_emb = self._area_emb(data['area'])
        block_emb = self._block_emb(data['block'])
        area_emb = (data['area_mask'].unsqueeze(2) * area_emb).sum(dim=1) / data['area_mask'].sum(dim=1).clamp(min=1).unsqueeze(1)
        block_emb = (data['block_mask'].unsqueeze(2) * block_emb).sum(dim=1) / data['block_mask'].sum(dim=1).clamp(min=1).unsqueeze(1)
        all_emb = torch.stack([act_1_emb, act_2_emb, act_3_emb, area_emb, block_emb])
        all_emb = self._W * all_emb
        all_emb = all_emb.sum(dim=0)
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
    def __init__(self, bank: GroupBank):
        super().__init__()
        self._group_vec = GroupVectorizer(bank)
        self._all_group_index = nn.Parameter(torch.LongTensor(list(range(len(bank.get_all_ids())))), requires_grad=False)
        self._all_group_data = nn.ParameterDict({k: nn.Parameter(v, requires_grad=False) for k, v in bank.get_model_data(bank.get_all_ids()).items()})
        self._cls = nn.Parameter(torch.empty([1, 1, EMBED_DIM]))
        nn.init.normal_(self._cls)
        self._pos_enc = nn.Embedding(512, EMBED_DIM)
        cfg = RobertaConfig(
            hidden_size=EMBED_DIM,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=EMBED_DIM * 4
        )
        self._encoder = RobertaEncoder(cfg)
        self._pool = RobertaPooler(cfg)
        self._loss = ArcFace()
        # self._loss = TripletMarginLoss(distance=CosineSimilarity())

    def forward(self, group_seq, group_pos, group_mask, group_target):
        all_group_vec = self._group_vec(self._all_group_data)
        self._loss.W = all_group_vec.T
        self._loss.num_classes = all_group_vec.shape[0]
        x = all_group_vec[group_seq]
        # pos_ids = torch.arange(curr_vec.shape[1] - 1, -1, -1)
        x = x + self._pos_enc(group_pos)
        x = torch.cat([self._cls.repeat(x.shape[0], 1, 1), x], dim=1)
        group_mask = torch.cat([torch.ones([group_mask.shape[0], 1], dtype=group_mask.dtype, device=group_mask.device), group_mask], dim=1)
        group_mask = get_extended_attention_mask(group_mask, dtype=x.dtype)
        x = self._encoder(x, attention_mask=group_mask)
        x = self._pool(x.last_hidden_state)
        loss = self._loss(x, group_target)
        return loss, self._loss.get_logits(x)


class MyTrainable(XZTrainable):
    def __init__(self, bank: GroupBank):
        self._bank = bank

    def step(self, context: BaseContext, data: DataType) -> Tuple[Tensor, ModelOutputsType]:
        loss, logits = context.model(data['group_sequence'], data['group_pos'], data['group_mask'], data['group_target'])
        return loss, {
            'loss': loss,
            'target': data['group_target'],
            'proba': torch.softmax(logits, dim=1)
        }

    def create_metrics(self, context_type: ContextType) -> Dict[str, Metric]:
        return {
            'loss': MeanMetric(),
            'accuracy_top1': Accuracy('multiclass', num_classes=self._bank.group_num()),
            'accuracy_top5': Accuracy('multiclass', num_classes=self._bank.group_num(), top_k=5),
            'accuracy_top10': Accuracy('multiclass', num_classes=self._bank.group_num(), top_k=10)
        }

    def update_metrics(self, context_type: ContextType, model_outputs: Dict[str, List], metrics: Dict[str, Metric]):
        metrics['loss'].update(model_outputs['loss'])
        metrics['accuracy_top1'].update(model_outputs['proba'], model_outputs['target'])
        metrics['accuracy_top5'].update(model_outputs['proba'], model_outputs['target'])
        metrics['accuracy_top10'].update(model_outputs['proba'], model_outputs['target'])
