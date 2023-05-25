import random
import re
import warnings
from typing import Tuple, Set, Dict, List, Optional, FrozenSet

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
from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaPooler
from xztrainer import XZTrainable, ContextType, BaseContext, DataType, ModelOutputsType

from dedformer.arcface import ArcFace
from dedformer.t5 import T5ForConditionalGeneration

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
        group_by_cols = ['направление 1', 'направление 2', 'направление 3', 'округ площадки', 'район площадки']
        self._macro_groups = {i: {'ids': set(x[1].index), 'features': {k: v for k, v in zip(group_by_cols, x[0])}} for i, x in enumerate(df.groupby(group_by_cols))}
        self._group_to_macro_group = {idx: index for index, itms in self._macro_groups.items() for idx in itms['ids']}

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
    def __init__(self, attend: Dict, bank: GroupBank, is_train: bool, dummy: bool):
        self._bank = bank
        self._attend = attend
        self._attend_indices = [k for k in attend.keys()]
        self._is_train = is_train
        self._dummy = dummy

    def __getitem__(self, index):
        item = self._attend[self._attend_indices[index]]
        group_nums = item['group_sequence']
        # if self._is_train:
        #     min_length = 1
        #     group_start = random.randint(0, len(group_nums) - min_length)
        #     group_end = group_start + random.randint(group_start, len(group_nums) - 1)
        #     group_nums = group_nums[group_start:group_end + 1]
        group_nums = group_nums[::-1][:500][::-1]
        if self._dummy:
            i = random.randint(0, self._bank.macro_group_num() - 1)
            group_nums = [i for _ in range(random.randint(3, 40))]
        return {
            'group_sequence': torch.LongTensor(group_nums[:-1]),
            'group_sequence_out': torch.LongTensor(group_nums),
            'group_target': torch.scalar_tensor(group_nums[-1], dtype=torch.long),
            'group_pos': torch.LongTensor(list(range(len(group_nums)))[::-1]),
            'group_mask': torch.LongTensor([1 for _ in range(len(group_nums))])
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
        self._all_group_index = nn.Parameter(torch.LongTensor(list(range(bank.num_macro_groups))), requires_grad=False)
        self._all_group_data = nn.ParameterDict({k: nn.Parameter(v, requires_grad=False) for k, v in bank.get_model_data(list(range(bank.num_macro_groups))).items()})
        self._cls = nn.Parameter(torch.empty([1, 1, EMBED_DIM]))
        self._sos = nn.Parameter(torch.empty([1, 1, EMBED_DIM]))
        nn.init.normal_(self._cls)
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
        # self._loss = TripletMarginLoss(distance=CosineSimilarity())

    def forward(self, group_seq, group_pos, group_mask, group_target, group_seq_out):
        all_group_vec = self._group_vec(self._all_group_data)
        x_dec = all_group_vec[group_seq]
        # pos_ids = torch.arange(curr_vec.shape[1] - 1, -1, -1)
        x_dec = x_dec + self._pos_enc(group_pos)
        x_dec = torch.cat([self._sos.repeat(x_dec.shape[0], 1, 1), x_dec], dim=1)
        x_enc = self._cls.repeat(x_dec.shape[0], 1, 1)
        enc_mask = torch.ones([x_enc.shape[0], 1], dtype=group_mask.dtype, device=group_mask.device)
        # group_mask = get_extended_attention_mask(group_mask, dtype=x.dtype)
        x = self._t5(input_embeds=x_enc, input_attention_mask=enc_mask, decoder_inputs_embeds=x_dec,
                     decoder_attention_mask=group_mask).logits
        loss, logits = self._loss(all_group_vec, x[group_mask == 1], group_seq_out[group_mask == 1])
        return loss, logits[group_mask.sum(dim=1).cumsum(0) - 1]


class MyTrainable(XZTrainable):
    def __init__(self, bank: GroupBank):
        self._bank = bank

    def step(self, context: BaseContext, data: DataType) -> Tuple[Tensor, ModelOutputsType]:
        loss, logits = context.model(data['group_sequence'], data['group_pos'], data['group_mask'], data['group_target'], data['group_sequence_out'])
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
