import torch
from functools import wraps
from torch import nn
from typing import Optional

from .data import ZScoreScaler
from .helper import set_device_recursive

__all__ = ['BJTrainer', 'BJDCRNNTrainer', 'BJSTGCNTrainer']


def train_wrapper(method):
    @wraps(method)
    def _impl(self, inputs, targets):
        attr, states, net = inputs

        if self.in_scalar:
            states = self.in_scalar.transform(states.to(self.device), 0)
        if self.out_scalar:
            targets = self.out_scalar.transform(targets.to(self.device), 0)

        predicts = method(self, (attr, states, net), targets)

        if self.out_scalar:
            return self.out_scalar.inverse_transform(predicts, 0)
        else:
            return predicts

    return _impl


def predict_wrapper(method):
    @wraps(method)
    def _impl(self, inputs):
        attr, states, net = inputs
        if self.in_scalar:
            states = self.in_scalar.transform(states.to(self.device), 0)

        predicts = method(self, (attr, states, net))

        if self.out_scalar:
            return self.out_scalar.inverse_transform(predicts, 0)
        else:
            return predicts

    return _impl


class TrainerBase(object):
    def __init__(self,
                 model: nn.Module,
                 loss: nn.Module,
                 device: torch.device,
                 optimizer,
                 in_scalar: ZScoreScaler,
                 out_scalar: ZScoreScaler,
                 max_grad_norm: Optional[float]):
        self.in_scalar = in_scalar
        self.out_scalar = out_scalar
        self.model = model.to(device)
        self.loss = loss.to(device)
        self.optimizer = optimizer
        self.clip = max_grad_norm
        self.device = device

    @train_wrapper
    def train(self, inputs, targets) -> torch.Tensor:
        raise NotImplementedError()

    @predict_wrapper
    def predict(self, inputs) -> torch.Tensor:
        raise NotImplementedError()

    def load_state_dict(self, model_state_dict, optimizer_state_dict):
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.model = self.model.to(self.device)
        set_device_recursive(self.optimizer.state, self.device)


class BJTrainer(TrainerBase):

    @train_wrapper
    def train(self, inputs, targets):
        self.model.train()
        self.optimizer.zero_grad()

        attr, states, net = inputs

        preds = self.model(attr.to(self.device), states.to(self.device), net.to(self.device), targets.to(self.device))

        loss = self.loss(preds, targets)
        loss.backward()
        if self.clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return preds

    @predict_wrapper
    def predict(self, inputs):
        self.model.eval()

        attr, states, net = inputs

        return self.model(attr.to(self.device), states.to(self.device), net.to(self.device))


class BJDCRNNTrainer(TrainerBase):
    def __init__(self, *args, **kwargs):
        super(BJDCRNNTrainer, self).__init__(*args, **kwargs)
        self.batch_seen = 0

    @train_wrapper
    def train(self, inputs, targets):
        self.batch_seen += 1

        self.model.train()
        self.optimizer.zero_grad()

        attr, states, net = inputs
        predicts = self.model(states.to(self.device), net.to(self.device), self.batch_seen, targets.to(self.device))

        loss = self.loss(predicts, targets.to(self.device))
        loss.backward()
        if self.clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return predicts

    @predict_wrapper
    def predict(self, inputs):
        self.model.eval()

        attr, states, net = inputs
        return self.model(states.to(self.device), net.to(self.device), self.batch_seen)


class BJSTGCNTrainer(TrainerBase):

    @train_wrapper
    def train(self, inputs, targets):
        self.model.train()
        self.optimizer.zero_grad()

        attr, states, net = inputs
        predicts = self.model(None, states.to(self.device), net.to(self.device), targets.to(self.device))

        loss = self.loss(predicts, targets.to(self.device))
        loss.backward()
        if self.clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return predicts

    @predict_wrapper
    def predict(self, inputs):
        self.model.eval()

        attr, states, net = inputs
        return self.model(None, states.to(self.device), net.to(self.device))
