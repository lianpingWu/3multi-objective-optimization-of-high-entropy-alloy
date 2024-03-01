# -*- coding: utf-8 -*-

import numpy as np
        
class defParameters_stru:
    
    def __init__(self, fraction):
        self.fraction = fraction / fraction.sum(axis=1, keepdims=1)
        self.fraction_num = self.fraction.shape[0]
        self.rad_list = [192, 163, 189, 194, 197] # radius, unit: [pm]
        self.VEC_list = [9, 10, 6, 8, 7] # valence electron concentration (containing d orbital electrons)
        self.H_list = [0, -4, -1, -5, -7, -2, -8, -1, 2, 0] # mix entropy
        
        
    def H(self):
        H = np.zeros(self.fraction_num)
        for idx in range(self.fraction_num):
            index = 0
            for loopi in range(5):
                for loopj in range(loopi + 1, 5):
                    H[idx] += 4 * self.H_list[index] * self.fraction[idx, loopi] * self.fraction[idx, loopj]
                    index += 1
        return H
        
        
    def VEC(self):
        VEC = np.zeros(self.fraction_num)
        for idx in range(self.fraction_num):
            for VEC_ele, frac in zip(self.VEC_list, self.fraction[idx, :]):
                VEC[idx] += frac * VEC_ele
        return VEC
        
        
    def delta_r(self):
        delta_r = np.zeros(self.fraction_num)
        for idx in range(self.fraction_num):
            r = 0 
            for rad, frac in zip(self.rad_list, self.fraction[idx,:]):
                r += rad * frac
            
            temp = 0
            for rad, frac in zip(self.rad_list, self.fraction[idx,:]):
                temp += frac * (1 - rad / r) ** 2
            
            delta_r[idx] = np.sqrt(temp)
            
        return delta_r * 100
        
        
def gen_samples(sample_num):
    fraction_norm = np.zeros([sample_num, 5])
    index = 0
    while index != sample_num:
        fraction_norm_temp = np.random.random([1, 5])
        fraction_norm_temp /= np.sum(fraction_norm_temp)
        
        if defParameters_stru(fraction_norm_temp).H() <= 7 and \
        defParameters_stru(fraction_norm_temp).H() >= -22 and \
        defParameters_stru(fraction_norm_temp).VEC() >= 8 and \
        defParameters_stru(fraction_norm_temp).delta_r() <= 6.6:
            
            fraction_norm[index,:] += fraction_norm_temp.reshape(5,)
            index += 1
            
    return fraction_norm
