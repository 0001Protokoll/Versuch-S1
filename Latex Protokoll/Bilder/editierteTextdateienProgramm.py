# -*- coding: utf-8 -*-

"""
Created on Sun May  6 17:42:22 2018

@author: David
"""

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#### wenn mehr dateien geladen werden sollen, einfach Liste erweitern ohne großen Umstand schön und sehr clever! ---verstehen wohl blos einige nicht ...
names = [
        'CO.txt',
        'CO30.txt',
        'HCL.txt',
        'HCL30.txt',
        'hclco.txt',
        'hclco30.txt',
        'CCSD CO 60.txt',
        'CCSD CO 30.txt',        
        ]

def getData(name):
    
    return np.genfromtxt(str(name), dtype=float, comments='#')
                         
data = [getData(i) for i in names]

####Überschrift wird als Marker für die erhaltenen Infos verwendet ... 
Überschrift=np.array([                                                          #n
        'CO',                                                #0
        'HCl',                                               #1
        'HCl CSSD (SCF)',                                         #2
        'HCl CSSD (MP2)',                                         #3
        'HCl CSSD (MP3)',                                         #4
        'CO CSSD (SCF)',                                             #5
        'CO CSSD (MP2)',                                             #6
        'CO CSSD (MP3)', 
        ])
###Latex erlaubt keine leerstellen. Hier werden die Speichernamen der Bilder festgelegt.
Speicher=np.array([                 
         'CO_b3lyp',                                                     #0
        'HCl_b3lyp',                                                     #1
        'HCl_CSSD(SCF)',                                                   #2
        'HCl_CSSD(MP2)',                                                   #3
        'HCl_CSSD(MP3)',                                                   #4
        'CO_CSSD(SCF)',                                                    #5
        'CO_CSSD(MP2)',                                                    #6
        'CO_CSSD(MP3)',
        ])


Disso=''
HCl=''
CO=''
print (data[4]) 

for n in range(2):
    a=data[n]
    b=data[n+2]
    for i in range(len(a)):    
        Disso+=str(a[i,1])+'&'+str(a[i,2])+'&'+str(b[i,1])+'&'+str(b[i,2])+'\\\\'+'\n'
        
for n in range(4,6):
    a=data[n]
    for i in range(len(a)): 
        
        HCl+=str(a[i,1])+'&'+str(a[i,2])+'&'+str(a[i,3])+'&'+str(a[i,4])+'\\\\'+'\n'
       

       
for n in range(6,8):
    a=data[n]
    
    for i in range(len(a)):    
        CO+=str(a[i,1])+'&'+str(a[i,2])+'&'+str(a[i,3])+'&'+str(a[i,4])+'\\\\'+'\n'
##### und noch der Save von diss.k. und R_0
   

f = open('workfile.txt', 'w')
f.write(Disso)
f.close()

b = open('HCLCCSD.txt', 'w')
b.write(HCl)
b.close()

c = open('COCCSD.txt', 'w')
c.write(CO)
c.close()