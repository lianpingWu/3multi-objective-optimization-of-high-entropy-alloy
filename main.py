# -*- coding: utf-8 -*-

import sys
import numpy as np
from func.data_loader import data_loader
from func.model import AlloyFSL
from func.train_func import train_func, setup_seed
from func.utils import mkdir
from func.feature_selection import define_list, eval_epoch


def main_train(flag, use_list, samples_path, times, save_path):
    inputs_num = len(use_list)
    train_dataloader, valid_dataloader, support_set = data_loader(flag, use_list, 2, samples_path)
    
    for index in range(times):
        container = train_func(
            inputs_num=inputs_num, 
            model=AlloyFSL(support_set, inputs_num),
            trainset_dataloader=train_dataloader, 
            validset_dataloader=valid_dataloader,
            support_set=support_set,
            save_file=save_path + "model_" + str(index) + ".pkl"
            )

        train_loss_record, valid_loss_record, _ = container.model_train()  # train a model
        np.savetxt(save_path + "train_loss_record_" + str(index) + ".txt", train_loss_record)
        np.savetxt(save_path + "valid_loss_record_" + str(index) + ".txt", valid_loss_record)
        np.savetxt(save_path + "use_list.txt", use_list, fmt="%d")
    

if __name__ == "__main__":
    setup_seed(512)
    samples_path = "./dataset/"
    flag = str(sys.argv[1])
    
    if flag == "YM":
        remove_list = [8]
    else:
        remove_list = [7]
        
    for epoch in range(1, 20):
        inputs_list = define_list(flag, remove_list)
        
        for index, use_list in enumerate(inputs_list):
            save_path = "./model_save/" + flag + "/" + str(epoch) + "/" + str(index) + "/"
            mkdir(save_path, clear=True)
            main_train(flag, use_list, samples_path, 10 ,save_path)

        remove_item = eval_epoch(flag, epoch, remove_list)
        print("epoch:", epoch, "=" * 5, remove_item)
        
        if remove_item == 0:
            break
        
        remove_list.append(remove_item)
