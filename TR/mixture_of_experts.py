import torch
import torch.optim as optim
from typing import Any, Callable, List, Optional
from torch import Tensor, nn
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.distributions.normal import Normal
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import time
import math


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out


class MoE(nn.Module):
    def __init__(self, input_channel, output_channel, input_size, num_experts, hidden_size, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_channel = output_channel
        self.input_channel = input_channel
        self.hidden_size = hidden_size
        self.k = k
        self.experts = nn.ModuleList(
            [Expert2(self.input_channel, self.output_channel, self.hidden_size) for i in range(self.num_experts)])

        self.w_gate = nn.Parameter(torch.zeros(input_size[0], input_size[1], input_channel, num_experts),
                                   requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size[0], input_size[1], input_channel, num_experts),
                                    requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

        self.ave = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.output_channel, 2)

    def cv_squared(self, x):

        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = torch.einsum("bchw,whce->be", x, self.w_gate)
        # clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            # raw_noise_stddev = x @ self.w_noise
            raw_noise_stddev = torch.einsum("bchw,whce->be", x, self.w_noise)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)  
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)

        expert_inputs = dispatcher.dispatch(x)

        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss


class Expert2(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

        self.input_size = input_size

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = self.fc1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out


if __name__ == "__main__":
    net = MoE(input_channel=1280, input_size=[1, 1], hidden_size=128, output_channel=2, num_experts=3,
              noisy_gating=True, k=2)
    a = torch.randn(size=(3000, 1280, 1, 1)).cuda()
    net = net.cuda()
    out, _ = net(a)

    print(out.shape)
