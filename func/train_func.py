# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import mean_absolute_error


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def train(model, loader, optimizer, loss_func):
    model.train()
    for coord, labels in loader:
        outputs = model(coord)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return model


@torch.no_grad()
def valid(model, train_loader, valid_loader, loss_func):
    model.eval()
    train_loss = 0
    for coord, labels in train_loader:
        outputs = model(coord)
        loss = loss_func(outputs, labels)
        train_loss += loss.item() / len(train_loader)
        
    valid_loss = 0
    for coord, labels in valid_loader:
        outputs = model(coord)
        loss = loss_func(outputs, labels)
        valid_loss += loss.item() / len(valid_loader)
        
    return train_loss, valid_loss


class train_func:
    def __init__(self, inputs_num, model, trainset_dataloader, 
                 validset_dataloader, support_set, save_file):
        
        self.inputs_num = inputs_num
        self.epochs = 300
        self.model = model
        self.trainset_dataloader = trainset_dataloader
        self.validset_dataloader = validset_dataloader
        self.support_set = support_set
        self.save_file = save_file
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=100,
            gamma=0.5
            )
        
        self.calc_MSE_loss = nn.MSELoss(reduction="mean")
        
        
    def model_train(self):
        flag_loss = 1.0e20
        train_loss_record, valid_loss_record = [], []
        for _ in range(self.epochs):
            self.model = train(
                model=self.model, 
                loader=self.trainset_dataloader, 
                optimizer=self.optimizer, 
                loss_func=self.calc_MSE_loss
                )
            
            self.scheduler.step()
            train_loss, valid_loss = valid(
                model=self.model, 
                train_loader=self.trainset_dataloader, 
                valid_loader=self.validset_dataloader, 
                loss_func=self.calc_MSE_loss
                )
            
            train_loss_record.append(train_loss)
            valid_loss_record.append(valid_loss)
            
            if train_loss < flag_loss:
                flag_loss = train_loss
                torch.save(obj=self.model.state_dict(), f=self.save_file)
                
        return train_loss_record, valid_loss_record, self.model