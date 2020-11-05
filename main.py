import torch

from models.lstm import FCLSTM
from utils.bj_dataset import bj_collate_fn, get_datasets
from utils.data import get_dataloaders
from utils.loss import get_loss
from utils.train import train_model, BJTrainer, get_optimizer, test_model

datasets = get_datasets()

model = FCLSTM(12, 12, 64, 2, 3, 2)

trainer = BJTrainer(
    model,
    loss=get_loss('MaskedMAELoss'),
    device=torch.device('cuda:0 '),
    optimizer=get_optimizer('Adam', model.parameters(), lr=0.0001),
    max_grad_norm=5
)
dls = get_dataloaders(datasets, batch_size=256, collate_fn=bj_collate_fn)
train_model(
    dls,
    folder='run',
    trainer=trainer,
    scheduler=None,
    epochs=100,
    early_stop_steps=None
)
test_model(
    dls['test'],
    trainer=trainer,
    folder='run',
)
