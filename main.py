import torch

from models import *
from utils.bj_dataset import bj_collate_fn, get_datasets
from utils.data import get_dataloaders, ZScoreScaler
from utils.helper import get_optimizer
from utils.loss import get_loss
from utils.train import train_model, test_model
from utils.trainers import *


def profile_data_loading():
    datasets = get_datasets()
    dls = get_dataloaders(datasets, batch_size=256, collate_fn=bj_collate_fn)
    from tqdm import tqdm
    for phase in ['train', 'val', 'test']:
        for _ in tqdm(dls[phase]):
            pass


def main():
    datasets = get_datasets(in_dims=('speed', 'available', 'total', 'speed_ha'), out_dims=('available',))
    dls = get_dataloaders(datasets, batch_size=16, collate_fn=bj_collate_fn)

    model = Ours(n_in=4, n_out=1)
    save_folder = 'ours_avail'

    trainer = BJTrainer(
        model,
        loss=get_loss('MaskedMAELoss'),
        device=torch.device('cuda:2'),
        optimizer=get_optimizer('Adam', model.parameters(), lr=0.001),
        in_scalar=ZScoreScaler(
            mean=torch.tensor([34.71207, 0.55837995, 1.454227, 35.422764, 0.57980937, 1.4051558], dtype=torch.float32),
            std=torch.tensor([11.989664, 0.28689522, 0.5432855, 9.341317, 0.15121026, 0.4632336], dtype=torch.float32)
        ),
        out_scalar=ZScoreScaler(
            mean=torch.tensor([0.55837995], dtype=torch.float32),
            std=torch.tensor([0.28689522], dtype=torch.float32)
        ),
        max_grad_norm=5
    )
    train_model(
        dls,
        folder=save_folder,
        trainer=trainer,
        scheduler=None,
        epochs=100,
        early_stop_steps=10
    )
    test_model(
        dls['test'],
        trainer=trainer,
        folder=save_folder,
    )


if __name__ == '__main__':
    # profile_data_loading()
    main()
