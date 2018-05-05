# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:07:01 2018

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
Überschrift=np.array([
        'CO',
        'CO30',
        'HCL',
        'HCL30',
        'hclco',
        'ander '
        ])



n=0

while n<= len(Datei)-1:
    fig = plt.figure()
    filename = Datei[n]
    print (filename)
    plt.plot(filename[:,1],filename[:,2],'x')
    plt.title(Überschrift[n])
       
    plt.xlabel("Radius r")
    plt.ylabel("Energie")
    n=n+1
    plt.savefig('Plot der Datei%d.png'%n, dpi=fig.dpi)
    plt.show()
    
   
    
    
