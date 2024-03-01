# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class f_model(nn.Module):
    def __init__(self, inputs_num):
        super(f_model, self).__init__()
        self.inputs_num = inputs_num
        self.feature_net = nn.Sequential(
            nn.Conv1d(2, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1),
            nn.ReLU()
            )

    def forward(self, x):
        return self.feature_net(x).view(-1, self.inputs_num)


class AlloyFSL(nn.Module):
    def __init__(self, support_set, inputs_num):
        super(AlloyFSL, self).__init__()
        self.support_set = support_set
        self.f_net_0 = f_model(inputs_num)
        self.f_net_1 = f_model(inputs_num)
        self.f_net_2 = f_model(inputs_num)
        self.f_net_3 = f_model(inputs_num)
        self.f_net_4 = f_model(inputs_num)

        self.outputs_channel = inputs_num * 5
        self.regression = nn.Sequential(
            nn.Linear(self.outputs_channel, 32),
            nn.ReLU(),
            nn.Linear(32, 32 * 2),
            nn.ReLU(),
            nn.Linear(32 * 2, 1)
            )

    def forward(self, x):
        batch_size = x.shape[0]
        net_0 = self.f_net_0(torch.cat([x, self.support_set[0:1,:,:].repeat(batch_size, 1, 1)], dim=1))
        net_1 = self.f_net_1(torch.cat([x, self.support_set[1:2,:,:].repeat(batch_size, 1, 1)], dim=1))
        net_2 = self.f_net_2(torch.cat([x, self.support_set[2:3,:,:].repeat(batch_size, 1, 1)], dim=1))
        net_3 = self.f_net_3(torch.cat([x, self.support_set[3:4,:,:].repeat(batch_size, 1, 1)], dim=1))
        net_4 = self.f_net_4(torch.cat([x, self.support_set[4:5,:,:].repeat(batch_size, 1, 1)], dim=1))
        net = torch.cat([net_0, net_1, net_2, net_3, net_4], dim=1)
        outputs = self.regression(net)
        return outputs
    
    
if __name__ == "__main__":
    inputs = torch.randn([10, 1, 17])
    support_set = torch.randn([5, 1, 17])
    inputs_num = 17
    model = AlloyFSL(support_set, inputs_num)
    outputs = model(inputs)
    print(outputs.shape)
    
    
    