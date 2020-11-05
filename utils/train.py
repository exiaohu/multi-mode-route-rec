import copy
import json
import os
import pickle
import shutil
import time
from typing import Optional, List

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .evaluate import evaluate



def train_model(
        data_loaders,
        folder: str,
        trainer,
        epochs: int,
        early_stop_steps: Optional[int],
        scheduler=None,
        use_checkpoint=False,
):
    folder = os.path.join('run', folder)
    save_path = os.path.join(folder, 'best_model.pkl')
    if use_checkpoint and os.path.exists(save_path):
        save_dict = torch.load(save_path)

        trainer.load_state_dict(save_dict['model_state_dict'], save_dict['optimizer_state_dict'])

        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch'] + 1
    else:
        shutil.rmtree(folder, ignore_errors=True)
        save_dict = dict()
        best_val_loss = float('inf')
        begin_epoch = 0

    phases = data_loaders.keys()

    writer = SummaryWriter(folder)

    since = time.perf_counter()

    print(trainer.model)
    print(f'Trainable parameters: {get_number_of_parameters(trainer.model)}.')

    try:
        for epoch in range(begin_epoch, begin_epoch + epochs):

            running_metrics = dict()
            for phase in phases:

                steps, predicts, targets = 0, list(), list()
                for x, y in tqdm(data_loaders[phase], f'{phase.capitalize():5} {epoch}'):
                    targets.append(y.numpy().copy())
                    if phase == 'train':
                        y_ = trainer.train(x, y)
                    else:
                        with torch.no_grad():
                            y_ = trainer.predict(x)

                    predicts.append(y_.detach().cpu().numpy())

                    # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
                    torch.cuda.empty_cache()

                # 性能
                running_metrics[phase] = evaluate(np.concatenate(predicts), np.concatenate(targets))

                if phase == 'val':
                    if running_metrics['val']['loss'] < best_val_loss:
                        best_val_loss = running_metrics['val']['loss']
                        save_dict.update(
                            model_state_dict=copy.deepcopy(trainer.model.state_dict()),
                            epoch=epoch,
                            best_val_loss=best_val_loss,
                            optimizer_state_dict=copy.deepcopy(trainer.optimizer.state_dict()))
                        save_model(save_path, **save_dict)
                        print(f'A better model at epoch {epoch} recorded.')
                    elif early_stop_steps is not None and epoch - save_dict['epoch'] > early_stop_steps:
                        raise ValueError('Early stopped.')

            loss_dict = {f'{phase} loss': running_metrics[phase].pop('loss') for phase in phases}
            print(loss_dict)

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(loss_dict['train'])
                else:
                    scheduler.step()

            writer.add_scalars('Loss', loss_dict, global_step=epoch)
            for metric in running_metrics['train'].keys():
                for phase in phases:
                    writer.add_scalars(f'{metric}', {f'{phase}': running_metrics[phase][metric]}, global_step=epoch)
    except (ValueError, KeyboardInterrupt) as e:
        print(e)
    finally:
        time_elapsed = time.perf_counter() - since

        writer.flush()
        print(f"cost {time_elapsed} seconds")
        print(f'The best adaptor and model of epoch {save_dict["epoch"]} successfully saved at `{save_path}`')


def test_model(
        dataloader,
        trainer,
        folder: str):
    folder = os.path.join('run', folder)
    saved_path = os.path.join('run', folder, 'best_model.pkl')

    saved_dict = torch.load(saved_path)
    trainer.model.load_state_dict(saved_dict['model_state_dict'])

    predictions, running_targets = list(), list()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, 'Test model'):
            running_targets.append(targets.numpy().copy())
            predicts = trainer.predict(inputs)
            predictions.append(predicts.cpu().numpy())

    # 性能
    predictions, running_targets = np.concatenate(predictions), np.concatenate(running_targets)
    scores = evaluate(predictions, running_targets)
    scores.pop('loss')
    print('test results:')
    print(json.dumps(scores, cls=JsonEncoder, indent=4))
    with open(os.path.join(folder, 'test-scores.json'), 'w+') as f:
        json.dump(scores, f, cls=JsonEncoder, indent=4)

    np.savez(os.path.join(folder, 'test-results.npz'), predictions=predictions, targets=running_targets)


def save_model(path: str, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


def get_number_of_parameters(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def get_scheduler(name, optimizer, **kwargs):
    return getattr(optim.lr_scheduler, name)(optimizer, **kwargs)


def get_optimizer(name: str, parameters, **kwargs):
    return getattr(optim, name)(parameters, **kwargs)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def set_device_recursive(var, device):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = set_device_recursive(var[key], device)
        else:
            try:
                var[key] = var[key].to(device)
            except AttributeError:
                pass
    return var


def set_requires_grad(models: List[nn.Module], required: bool):
    for model in models:
        for param in model.parameters():
            param.requires_grad_(required)


def get_regularization(model: nn.Module, weight_decay: float, p: float = 2.0):
    weight_list = list(filter(lambda item: 'weight' in item[0], model.named_parameters()))
    return weight_decay * sum(torch.norm(w, p=p) for name, w in weight_list)
