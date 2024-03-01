# -*- coding: utf-8 -*-

import numpy as np


class defParameters:
    """
    This class is desgined for calculate the parameters of HEA inputs:
    The sort is Co, Ni, Cr, Fe, Mn
    This data comes from:
        1. https://chem.libretexts.org/Ancillary_Materials/Reference/Periodic_Table_of_the_Elements
        2. Melting point obtained from: https://periodictable.com/Elements/027/data.html
        3. H_list obtained from: Takeuchi, A., and Inoue, A. (2005). Classification of Bulk Metallic Glasses by Atomic Size
        Difference, Heat of Mixing and Period of Constituent Elements and Its Application to Characterization of the Main
        Alloying Element. MATERIALS TRANSACTIONS 46, 2817-2829.
    """
    
    def __init__(self, fraction):
        self.fraction = fraction / sum(fraction)
        self.rad_list = [192, 163, 189, 194, 197] # radius, unit: [pm]
        self.chi_list = [1.88, 1.91, 1.83, 1.66, 1.55] # Electronegativity
        self.VEC_list = [9, 10, 6, 8, 7] # valence electron concentration (containing d orbital electrons)
        self.E_list = [209, 200, 279, 211, 198] # Young's modulus [GPa]
        self.T_list = [1768, 1455, 1907, 1538, 1246] # unit: [°C]
        self.H_list = [0, -4, -1, -5, -7, -2, -8, -1, 2, 0] # mix entropy
        self.mu_list = [0.320, 0.312, 0.210, 0.291, 0.24] # Poisson ratio
        self.Ec_list = [4.39, 4.44, 4.10, 4.13, 2.92] # cohesive energy
        self.G_list = [75, 76, 115, 82, 76.4] # shear modulus [GPa]


    def delta_r(self):
        r_mean = np.mean(self.rad_list) # the mean value of radius
        delta_r = 0
        for rad, frac in zip(self.rad_list, self.fraction):
            delta_r += frac * (1 - rad / r_mean) ** 2
        return np.sqrt(delta_r) * 10


    def delta_chi(self):
        chi_mean = np.mean(self.chi_list) # the mean value of electronegativity
        delta_chi = 0
        for chi_ele, frac in zip(self.chi_list, self.fraction):
            delta_chi += frac * (chi_ele - chi_mean) ** 2
        return np.sqrt(delta_chi) * 10


    def VEC(self):
        VEC = 0
        for VEC_ele, frac in zip(self.VEC_list, self.fraction):
            VEC += frac * VEC_ele
        return VEC


    def delta_H(self):
        index, H = 0, 0
        for loopi in range(len(self.fraction)):
            for loopj in range(loopi + 1, len(self.fraction)):
                H += 4 * self.H_list[index] * self.fraction[loopi] * self.fraction[loopj]
                index += 1
        return H


    def delta_S(self):
        R, S = 8.3145, 0
        for frac in self.fraction:
            if frac == 0:
                S += 0
            else:
                S += frac * np.log(frac)
        return - R * S


    def omega(self):
        return np.mean(self.T_list) * (self.delta_S() / (np.abs(self.delta_H()) + 1e-8)) / 500


    def Lambda(self):
        return self.delta_S() / (self.delta_r() ** 2 + 1e-8)


    def DX(self):
        DX = 0
        for loopi in range(len(self.fraction)):
            for loopj in range(loopi, len(self.fraction)):
                DX += self.fraction[loopi] * self.fraction[loopj] * np.abs(self.chi_list[loopi] - self.chi_list[loopj])
        return DX * 50


    def Ec(self):
        EC = 0
        for frac, Ec_ele in zip(self.fraction, self.Ec_list):
            EC += frac * Ec_ele
        return EC
    

    def eta(self):
        G = self.G()
        eta = 0
        for frac, G_ele in zip(self.fraction, self.G_list):
            eta += (frac * 2 * (G_ele - G) / (G_ele + G)) / \
            (1 + 0.5 * np.abs((frac * 2 * (G_ele - G) / (G_ele + G))))
        return eta


    def Dr(self):
        Dr = 0
        for loopi in range(len(self.fraction)):
            for loopj in range(loopi, len(self.fraction)):
                Dr += self.fraction[loopi] * self.fraction[loopj] * np.abs(self.rad_list[loopi] - self.rad_list[loopj])
        return Dr


    def A(self):
        mu = np.mean(self.mu_list)
        A = self.G() * self.delta_r() * (1 + mu) * (1 - mu)
        return A


    def F(self): 
        return (2 * self.G()) / (1 - np.mean(self.mu_list)) / 10


    def G(self): 
        G = 0
        for frac, G_ele in zip(self.fraction, self.G_list):
            G += frac * G_ele
        return G / 10


    def delta_G(self):
        G = self.G()
        delta_G = 0
        for frac, G_ele in zip(self.fraction, self.G_list):
            delta_G += frac * (1 - G_ele / G) ** 2
        return np.sqrt(delta_G)


    def DG(self):
        DG = 0
        for loopi in range(len(self.fraction)):
            for loopj in range(loopi, len(self.fraction)):
                DG += self.fraction[loopi] * self.fraction[loopj] * np.abs(self.G_list[loopi] -self.G_list[loopj])
        return DG


    def mu(self):
        mu = np.mean(self.E_list) * self.delta_r()
        return 0.5 * mu / 10


    def dataset(self):
        props = np.array([[self.delta_r(), self.delta_chi(), self.VEC(), self.delta_H(), 
                           self.delta_S(), self.omega(), self.Lambda(), self.DX(), 
                           self.Ec(), self.eta(), self.Dr(), self.A(), self.F(), self.G(), 
                           self.delta_G(), self.DG(), self.mu()]])

        return np.hstack([props, np.array([self.fraction])])

    
class build_dataset:
    def __init__(self, target, data_path):
        self.samples = np.load(data_path)
        
        if target == "YS":
            self.index = 5
        elif target == "CRSS":
            self.index = 6


    def define_performance(self):
        performance = np.zeros([self.samples.shape[0], 1])
        for loopi in range(self.samples.shape[0]):
            performance[loopi,:] = self.data[loopi, self.index]
        return performance


    def define_samples(self): # define the dataset
        for loopi in range(self.fraction.shape[0]):
            if loopi == 0:
                samples_temp = defParameters(self.fraction[loopi]).dataset()
                samples = np.zeros([self.fraction.shape[0], 1, samples_temp.shape[1]])
                samples[loopi,:,:] = samples_temp
            else:
                samples[loopi,:,:] = defParameters(self.fraction[loopi]).dataset()
        return samples


    def build_train(self):
        X_train, Y_train = self.define_samples()[0:-1], self.define_performance()[0:-1]
        X_test, Y_test = self.define_samples()[-1], self.define_performance()[-1]
        X_test = np.expand_dims(X_test, axis=1) 
        Y_test = np.expand_dims(Y_test, axis=0)
        return X_train, Y_train, X_test, Y_test


    def build_support_set(self): 
        support_set_list = [0, 7, 9, 10, 18, 22, 23, 31, 32, 38]
        return self.define_samples()[support_set_list,:,:]


if __name__ == "__main__":
    sample_container = build_dataset(target="YS", data_path="./dataset/samples.npy")
#     X_train, Y_train, X_test, Y_test = sample_container.build_train()
#     support_set = sample_container.build_support_set()

#     print(X_train.shape, "X_train")
#     print(Y_train.shape, "Y_train")
#     print(X_test.shape, "X_test")
#     print(Y_test.shape, "Y_test")
