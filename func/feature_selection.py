# -*- coding: utf-8 -*-

import torch
import numpy as np
from func.model import AlloyFSL
from func.data_loader import data_loader


def MAPE(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / y_true)


def define_list(flag, remove_list):
    
    # define the inputs list
    if flag == "YM":
        original_list = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]
        
    elif flag == "CRSS":
        original_list = [1, 2, 4, 5, 6, 7, 8, 9, 10, 14]
        
    if remove_list != []:
        for element in remove_list:
            original_list.remove(element)
        
    inputs_list = []
    for loopi in range(len(original_list) + 1):
        if loopi == 0:
            inputs_list.append(original_list + [17, 18, 19, 20, 21])
        else:
            inputs_list.append(
                [_ for _ in original_list if _ is not original_list[loopi - 1]]  + [17, 18, 19, 20, 21]
                )
        
    return inputs_list


@torch.no_grad()
def valid_epoch(model, train_loader, valid_loader):
    model.eval()
    train_pred = np.zeros([len(train_loader)])
    train_true = np.zeros_like(train_pred)
    index = 0
    for coords, labels in train_loader:
        train_pred[index] = model(coords).item()
        train_true[index] = labels.item()
        index += 1
        
    valid_pred = np.zeros([len(valid_loader)])
    valid_true = np.zeros_like(valid_pred)
    index = 0
    for coords, labels in valid_loader:
        valid_pred[index] = model(coords).item()
        valid_true[index] = labels.item()
        index += 1
        
    train_MAPE = MAPE(y_true=train_true, y_pred=train_pred)
    valid_MAPE = MAPE(y_true=valid_true, y_pred=valid_pred)
    
    return train_MAPE, valid_MAPE


def eval_epoch(flag, epoch, remove_list):
    inputs_list = define_list(flag, remove_list)

    with torch.no_grad():
        valid_MAPE_record, train_MAPE_record = [], []
        for model_num, use_list in enumerate(inputs_list):
            model_path = "./model_save/" + flag + "/" + str(epoch) + "/" + str(model_num) + "/"
            inputs_num = len(use_list)
            train_dataloader, valid_dataloader, support_set = data_loader(flag, use_list, 1, "./dataset/")

            train_MAPE, valid_MAPE = 0, 0
            for index in range(10):
                model = AlloyFSL(support_set, inputs_num)
                model.load_state_dict(torch.load(model_path + "model_" + str(index) + ".pkl"))
                model.eval()
                train_MAPE_temp, valid_MAPE_temp = valid_epoch(model, train_dataloader, valid_dataloader)
                train_MAPE += train_MAPE_temp / 10
                valid_MAPE += valid_MAPE_temp / 10

            train_MAPE_record.append(train_MAPE)
            valid_MAPE_record.append(valid_MAPE)
    
    remove_index = np.where(valid_MAPE_record == min(valid_MAPE_record))[0][0] - 1
    remove_item = inputs_list[0][remove_index]
    
    return remove_item