#!/bin/bash
clear

for target in "YM" "CRSS"
    do
        nohup python -u train_model.py $target >running_$target.log 2>&1 &
    done

exit 0