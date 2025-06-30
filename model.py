import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tanh


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    def __init__(self, A3C_args, BTC_args):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(BTC_args.state_length, A3C_args.N_nodes_per_layer)
        self.fc2 = nn.Linear(A3C_args.N_nodes_per_layer, A3C_args.N_nodes_per_layer)
        self.lstm = nn.LSTMCell(A3C_args.N_nodes_per_layer, A3C_args.N_nodes_per_layer)
        num_outputs = BTC_args.n_action
        self.critic_linear1 = nn.Linear(A3C_args.N_nodes_per_layer, 1)
        self.critic_linear2 = nn.Linear(A3C_args.N_nodes_per_layer, 1)
        self.actor_linear1 = nn.Linear(A3C_args.N_nodes_per_layer, num_outputs)
        self.actor_linear2 = nn.Linear(A3C_args.N_nodes_per_layer, 1)


        self.apply(weights_init)
        self.actor_linear1.weight.data = normalized_columns_initializer(
            self.actor_linear1.weight.data, 0.01)
        self.actor_linear1.bias.data.fill_(0)
        self.actor_linear2.weight.data = normalized_columns_initializer(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear1.weight.data = normalized_columns_initializer(
            self.critic_linear1.weight.data, 1.0)
        self.critic_linear1.bias.data.fill_(0)
        self.critic_linear2.weight.data = normalized_columns_initializer(
            self.critic_linear2.weight.data, 1.0)
        self.critic_linear2.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear1(x), self.critic_linear2(x), self.actor_linear1(x), self.actor_linear2(x), (hx, cx)
