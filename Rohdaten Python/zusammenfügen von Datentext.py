# -*- coding: utf-8 -*-
"""
Created on Fri May  4 09:40:39 2018

@author: David
"""

import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


Datei1 = np.genfromtxt("CO.txt",dtype=float, comments='#')
Datei2 = np.genfromtxt("CO30.txt",dtype=float,comments='#')
Datei3 = np.genfromtxt("HCL.txt",dtype=float,comments='#')
Datei4 = np.genfromtxt("HCL30.txt",dtype=float,comments='#')
Datei5 = np.genfromtxt("hclco.txt",dtype=float,comments='#')
Datei6 = np.genfromtxt("hclco30.txt",dtype=float,comments='#')

Datei=np.array([
        Datei1, 
        Datei2,
        Datei3,
        Datei4,
        Datei5,
        Datei6        
        ])



Spalte1=np.append(Datei3[:,1],Datei4[:,1])
Spalte2=np.append(Datei3[:,2],Datei4[:,2])

zusammenHCL= np.c_[Spalte1,Spalte2]

np.savetxt('HCLzusammen.txt',zusammen)



Spalte1=np.append(Datei5[:,1],Datei6[:,1])
Spalte2=np.append(Datei5[:,2],Datei6[:,2])

zusammenhclco= np.c_[Spalte1,Spalte2]

np.savetxt('hclcozusammen.txt',zusammen)