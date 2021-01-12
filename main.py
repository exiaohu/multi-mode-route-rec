import argparse

import torch

from models import *
from utils.bj_dataset import bj_collate_fn, get_datasets
from utils.data import get_dataloaders, ZScoreScaler
from utils.helper import get_optimizer
from utils.loss import get_loss
from utils.train import train_model, test_model
from utils.trainers import *

parser = argparse.ArgumentParser()
parser.add_argument('--save_folder', type=str, required=True, help='where to save the logs and results.')
parser.add_argument('--cuda', type=str, required=True, help='the torch device indicator, like "cuda:0".')
parser.add_argument('--loss', type=str, default='MaskedMAELoss', help='loss function, default as MaskedMAELoss.')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer name, default as `Adam`')
parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate.')
parser.add_argument('--max_grad_norm', type=float, default=5, help='clips gradient norm of an iterable of parameters.')
parser.add_argument('--epochs', type=int, default=100, help='the max number of epochs to train.')
parser.add_argument('--early_stop_steps', type=int, default=10, help='steps to early stop.')
parser.add_argument('--test', action='store_true', default=False, help='toggle the test mode.')
parser.add_argument('--resume', action='store_true', default=False, help='if to resume a trained model.')

args = parser.parse_args()


def main():
    datasets = get_datasets(in_dims=('speed', 'available', 'total', 'speed_ha'), out_dims=('speed',))
    dls = get_dataloaders(datasets, batch_size=16, collate_fn=bj_collate_fn)

    model = Ours(n_in=4, n_out=1)
    save_folder = args.save_folder

    trainer = BJTrainer(
        model,
        loss=get_loss(args.loss),
        device=torch.device(args.cuda),
        optimizer=get_optimizer(args.optimizer, model.parameters(), lr=args.lr),
        in_scalar=ZScoreScaler(
            mean=torch.tensor([34.71207, 0.55837995, 1.454227, 35.422764, 0.57980937, 1.4051558], dtype=torch.float32),
            std=torch.tensor([11.989664, 0.28689522, 0.5432855, 9.341317, 0.15121026, 0.4632336], dtype=torch.float32)
        ),
        out_scalar=ZScoreScaler(
            mean=torch.tensor([0.55837995], dtype=torch.float32),
            std=torch.tensor([0.28689522], dtype=torch.float32)
        ),
        max_grad_norm=args.max_grad_norm
    )
    if not args.test:
        train_model(
            dls,
            folder=save_folder,
            trainer=trainer,
            scheduler=None,
            epochs=args.epochs,
            early_stop_steps=args.early_stop_steps,
            use_checkpoint=args.resume
        )
    test_model(
        dls['test'],
        trainer=trainer,
        folder=save_folder,
    )


if __name__ == '__main__':
    main()
