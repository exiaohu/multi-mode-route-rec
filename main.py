import torch

from models import FCLSTM
from utils.bj_dataset import bj_collate_fn, get_datasets
from utils.data import get_dataloaders, ZScoreScaler
from utils.loss import get_loss
from utils.train import train_model, get_optimizer, test_model
from utils.trainers import BJTrainer

datasets = get_datasets()

model = FCLSTM()

trainer = BJTrainer(
    model,
    loss=get_loss('MaskedMAELoss'),
    device=torch.device('cuda:0'),
    optimizer=get_optimizer('Adam', model.parameters(), lr=0.001),
    scalar=ZScoreScaler(),
    max_grad_norm=5
)
dls = get_dataloaders(datasets, batch_size=256, collate_fn=bj_collate_fn)
train_model(
    dls,
    folder='fclstm',
    trainer=trainer,
    scheduler=None,
    epochs=100,
    early_stop_steps=10
)
test_model(
    dls['test'],
    trainer=trainer,
    folder='fclstm',
)
