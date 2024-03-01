# -*- coding: utf-8 -*-

import os 
import shutil

def mkdir(path, clear=False):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    
    if isExists:
        if clear:
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            pass
        
    else:
        os.makedirs(path)