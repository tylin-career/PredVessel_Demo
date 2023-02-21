import collections
import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import kaiming_uniform_, _calculate_fan_in_and_fan_out, uniform_
from torch.nn import MSELoss
from cseta.common.logger import logger


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_activation(name):
    def hook(model, input, output):
        zero_activation_y1 = float((output[:, 0] <= 1.0001).sum()) / output.size(0)
        zero_activation_y2 = float((output[:, 1] <= 1.0001).sum()) / output.size(0)
        print(
            "Zero Gradient Percentage y1 (%.2f) y2 (%.2f)"
            % (zero_activation_y1, zero_activation_y2)
        )

    return hook


class MAPELoss(torch.nn.CrossEntropyLoss):
    def __init__(self, device, config=None, reduction=None):
        super(MAPELoss, self).__init__(reduction)
        if 'standard' in config.target_transform:
            self.std_norm_const = torch.from_numpy(config.vars["std_norm_const"]).to(device)
            self.mean = torch.from_numpy(config.vars["target_mean"]).to(device)
            self.sd = torch.from_numpy(config.vars["target_sd"]).to(device)
        self.target_limit = config.target_limit
        self.with_uncertainty = config.with_uncertainty
        self.confidence_alpha = config.confidence_alpha
        self.target_transform = config.target_transform

    def combine_multiple_targets(self, y):
        if 'log' in self.target_transform:
            y = torch.exp(y) + 0
        if 'standard' in self.target_transform:
            y = (y - self.std_norm_const) * self.sd + self.mean
        y[:, 1] += y[:, 0]  # y2 = y1 + y2
        y[y != y] = 0  # fillna
        return y

    def forward(self, input, target, weight=None, l1_weight=None):
        if self.with_uncertainty:
            confidence = input[:,2:]
            combined_input = self.combine_multiple_targets(input[:,:2])
        else:
            combined_input = self.combine_multiple_targets(input)
        combined_target = self.combine_multiple_targets(target)
        # limited_target = torch.clamp(combined_target, min=3600, max=42*24*3600) #+ torch.clamp(combined_input, min=3600, max=18*24*3600)
        # mape = torch.abs((combined_target - combined_input) / limited_target)
        # losses = mape * weight  # [weight_pilot, weight_berth]
        smape = torch.abs(
            (combined_target[:, 0] - combined_input[:, 0])
            / (
                torch.clamp(
                    combined_target[:, 0],
                    min=self.target_limit[0],
                    max=self.target_limit[1],
                )
                + torch.clamp(
                    combined_input[:, 0],
                    min=self.target_limit[0],
                    max=self.target_limit[1],
                )
            )
        )
        mape = torch.abs(
            (combined_target[:, 1] - combined_input[:, 1])
            / torch.clamp(
                combined_target[:, 1],
                min=self.target_limit[0],
                max=self.target_limit[1],
            )
        )
        if self.with_uncertainty:
            losses = (torch.stack([smape, mape], dim=1) * confidence - torch.log(confidence) * self.confidence_alpha) * weight  # [weight_pilot, weight_berth]
        else:
            losses = (torch.stack([smape, mape], dim=1) * weight)  # [weight_pilot, weight_berth]
        loss = torch.sum(losses) / torch.sum(weight)
        errors = torch.abs((combined_target - combined_input)) * weight
        return loss, losses, errors


class L1Loss(torch.nn.CrossEntropyLoss):
    def __init__(self, device, config=None, reduction=None):
        super(L1Loss, self).__init__(reduction)
        if 'standard' in config.target_transform:
            self.std_norm_const = torch.from_numpy(config.vars["std_norm_const"]).to(device)
            self.mean = torch.from_numpy(config.vars["target_mean"]).to(device)
            self.sd = torch.from_numpy(config.vars["target_sd"]).to(device)
        self.with_uncertainty = config.with_uncertainty
        self.confidence_alpha = config.confidence_alpha
        self.target_transform = config.target_transform

    def combine_multiple_targets(self, y):
        if 'log' in self.target_transform:
            y = torch.exp(y) + 0
        if 'standard' in self.target_transform:
            y = (y - self.std_norm_const) * self.sd + self.mean
        y[:, 1] += y[:, 0]  # y2 = y1 + y2
        y[y != y] = 0  # fillna
        return y

    def forward(self, input, target, weight=None, l1_weight=None):
        # combined_input = input
        # combined_target = target
        if self.with_uncertainty:
            confidence = input[:,2:]
            combined_input = self.combine_multiple_targets(input[:,:2])
        else:
            combined_input = self.combine_multiple_targets(input)
        combined_target = self.combine_multiple_targets(target)
        mae = torch.abs(combined_target - combined_input)
        if self.with_uncertainty:
            mae = mae * confidence - torch.log(confidence) * self.confidence_alpha
        if l1_weight is not None:
            losses = mae * l1_weight  #normalized l1 loss
        else:
            losses = mae * weight  # [weight_pilot, weight_berth]
        loss = torch.sum(losses) / torch.sum(weight)
        errors = torch.abs((combined_target - combined_input)) * weight
        return loss, losses, errors


class RMSLELoss(torch.nn.CrossEntropyLoss):
    def __init__(self, device, config=None, reduction=None):
        super(RMSLELoss, self).__init__(reduction)
        if 'standard' in config.target_transform:
            self.std_norm_const = torch.from_numpy(config.vars["std_norm_const"]).to(device)
            self.mean = torch.from_numpy(config.vars["target_mean"]).to(device)
            self.sd = torch.from_numpy(config.vars["target_sd"]).to(device)
        self.mse = MSELoss()
        self.target_limit = config.target_limit
        self.target_transform = config.target_transform

    def combine_multiple_targets(self, y):
        if 'log' in self.target_transform:
            y = torch.exp(y) + 0
        if 'standard' in self.target_transform:
            y = (y - self.std_norm_const) * self.sd + self.mean
        y[:, 1] += y[:, 0]  # y2 = y1 + y2
        y[y != y] = 0  # fillna
        return y

    def forward(self, input, target, weight=None, l1_weight=None):
        combined_input = self.combine_multiple_targets(input)
        combined_target = self.combine_multiple_targets(target)
        square_error = (
            # torch.log(combined_input + 1) - torch.log(combined_target + 1)
            torch.log(torch.clamp(combined_input,min=self.target_limit[0],max=self.target_limit[1],)) - 
            torch.log(torch.clamp(combined_target,min=self.target_limit[0],max=self.target_limit[1],))
        ) ** 2
        losses = square_error * weight  # [weight_pilot, weight_berth]
        loss = torch.sqrt(torch.sum(losses) / torch.sum(weight))
        errors = torch.abs((combined_target - combined_input)) * weight
        return loss, losses, errors


class LinearZeroSlope(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearZeroSlope, self).__init__(in_features, out_features, bias)
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weight, a=0)
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            uniform_(self.bias, -bound, bound)


class ReluPlusOne(torch.nn.ReLU):
    def __init__(self, with_uncertainty=False, transform='standard normalization', inplace=False):
        super(ReluPlusOne, self).__init__()
        self.inplace = inplace
        self.with_uncertainty = with_uncertainty
        self.transform = transform
        
    def forward(self, input):
        if self.with_uncertainty:
            output = F.relu(input[:,:2], inplace=self.inplace) + 1 if 'standard' in self.transform else input[:,:2]
            confidence = torch.sigmoid(input[:,2:])
            return torch.cat([output, confidence], dim=1)
        else:
            return F.relu(input, inplace=self.inplace) + 1 if 'standard' in self.transform else input


class EarlyStopping:
    def __init__(self, patience, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        """
        self.patience = patience
        self.delta = delta
        self.early_stop = False
        self.counter = 0
        self.epoch = 0
        self.step = 0
        self.best_epoch = 0
        self.best_step = 0
        self.best_score = None
        self.best_model = None
        self.train_loss_min = np.Inf
        self.val_loss_min = np.Inf
        self.top_models = []
        self.num_top_models = 5
        self.avg_model = None

    def __call__(self, epoch, step, train_loss, val_loss, model):
        self.epoch = epoch
        self.step = step
        score = -val_loss
        msg = "step %d training loss: %f / validation loss: %f" % (
            step,
            train_loss,
            val_loss,
        )

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(train_loss, val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            msg += f" -- early stop counter {self.counter}/{self.patience}"
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(train_loss, val_loss, model)
            self.counter = 0
        logger.info(msg)

    def save_checkpoint(self, train_loss, val_loss, model):
        self.best_model = copy.deepcopy(model)
        self.train_loss_min = train_loss
        self.val_loss_min = val_loss
        self.best_epoch = self.epoch
        self.best_step = self.step
        self.top_models.append(
            {
                "model": self.best_model,
                "train_loss_min": self.train_loss_min,
                "val_loss_min": self.val_loss_min,
                "best_epoch": self.best_epoch,
                "best_step": self.best_step,
            }
        )
        if len(self.top_models) > self.num_top_models:
            self.top_models.pop(0)

    def average_top_models(self):
        params_dict = {}
        avg_model = self.best_model
        for model_id, learner in enumerate(self.top_models):
            for key, value in learner["model"].state_dict().items():
                if key not in params_dict:
                    params_dict[key] = value
                else:
                    params_dict[key] += value
        params_dict = {
            key: value / self.num_top_models for key, value in params_dict.items()
        }
        avg_model.load_state_dict(params_dict)
        return avg_model

    def get_summary(self):
        return {
            "best_epoch": self.best_epoch,
            "step": self.best_step,
            "train_loss": self.train_loss_min,
            "val_loss": self.val_loss_min,
        }
