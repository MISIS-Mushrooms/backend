import pickle

import torch
from madgrad import MADGRAD
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from xztrainer import XZTrainer, XZTrainerConfig, SchedulerType
from xztrainer.logger.tensorboard import TensorboardLoggingEngineConfig
from xztrainer.setup_helper import set_seeds, enable_tf32

from dedformer.model import GroupBank, GroupVectorizer, AllVectorizer, AttendanceDataset, AttendanceCollator, \
    MyTrainable

if __name__ == '__main__':
    enable_tf32()
    set_seeds(0x1337)
    bank = GroupBank('data/groups.csv')
    with open('data/attendance.pkl', 'rb') as f:
        attends = pickle.load(f)
    attend_users = list(attends.keys())
    attend_users_train, attend_users_val = train_test_split(attend_users, test_size=0.1, random_state=0x1337)
    attend_users_train, attend_users_val = set(attend_users_train), set(attend_users_val)
    attends_train, attends_val = {k: v for k, v in attends.items() if k in attend_users_train}, {k: v for k, v in attends.items() if k in attend_users_val}
    train_ds, val_ds = AttendanceDataset(attends_train, bank, is_train=True, dummy=False), AttendanceDataset(attends_val, bank, is_train=False, dummy=False)
    trainer = XZTrainer(XZTrainerConfig(
        experiment_name='t5-fix-4decoders-newmargin-32scae-0.05m-newg-2',
        batch_size=8,
        batch_size_eval=8,
        epochs=10,
        save_keep_n=3,
        optimizer=lambda mdl: MADGRAD(mdl.parameters(), lr=5e-4, weight_decay=0),
        scheduler=lambda opt, total_steps: get_linear_schedule_with_warmup(opt, int(total_steps * 0.2), total_steps),
        scheduler_type=SchedulerType.STEP,
        dataloader_num_workers=8,
        accumulation_batches=4,
        print_steps=50,
        save_steps=500,
        eval_steps=500,
        collate_fn=AttendanceCollator(),
        logger=TensorboardLoggingEngineConfig()
    ), AllVectorizer(bank), MyTrainable(bank))
    trainer.train(train_ds, val_ds)
