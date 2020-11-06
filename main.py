import torch

from models import FCLSTM
from utils.bj_dataset import bj_collate_fn, get_datasets
from utils.data import get_dataloaders, ZScoreScaler
from utils.loss import get_loss
from utils.train import train_model, get_optimizer, test_model
from utils.trainers import BJTrainer


def profile_data_loading():
    datasets = get_datasets()
    dls = get_dataloaders(datasets, batch_size=256, collate_fn=bj_collate_fn)
    from tqdm import tqdm
    for phase in ['train', 'val', 'test']:
        for _ in tqdm(dls[phase]):
            pass


def main():
    datasets = get_datasets()
    dls = get_dataloaders(datasets, batch_size=256, collate_fn=bj_collate_fn)

    model = FCLSTM()
    save_folder = 'fclstm'

    trainer = BJTrainer(
        model,
        loss=get_loss('MaskedMAELoss'),
        device=torch.device('cuda:0'),
        optimizer=get_optimizer('Adam', model.parameters(), lr=0.01),
        scalar=ZScoreScaler(),
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
    main()
