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
        'hclco30'
        ])


print (Datei1)
print (Datei2)
n=0

Test=np.append(Datei3[:,1],Datei4[:,1])
Test1=np.append(Datei3[:,2],Datei4[:,2])
tes=np.c_[Test,Test1]
#print (tes)
Temp= []
mini=1
abfrage=0
#plt.show()


def subtrak (ar,spalte):
        n=0
        mi=min(ar[:,spalte])
        print ("kleinster Wert", min(ar[:,spalte]))
        while n< len(ar):
            mi=ar[n,spalte]-min(ar[:,spalte])
            #print (mi)
            Temp.append(mi)
            n=n+1
        #Temp.appen
        print (Temp)
        
i = np.array(subtrak(tes,1))

#print(i)
#print(len(i))
        
#a=np.c_[Test1,Test] #baue spalten in neue MAtrix ein





#print (i)       
print (type(i))        
#print (type(tes))
#print (tes[:,0])
def al(x,y):     
        def fitFunc(x, V0, x0):
            return (V0/(x0**2))*(x-x0)**2
        
        fitParams, fitCovariances = curve_fit(fitFunc, x, y)
        
        print (fitParams)
        print (fitCovariances)
        #print (y)
        #plt.ylim(-461,-459)
        #plt.gca().set_ylim(top=)
        plt.plot(x, fitFunc(x, fitParams[0], fitParams[1]))
        plt.plot(x,y,'bo')
        plt.show()
        
        
#print (al(Test, tes[1,:]))

"""while n<= len(Datei)-1:
    fig = plt.figure()
    filename = Datei[n]
    #print (filename)
    plt.plot(filename[:,1],filename[:,2],'bo')
    n=n+1
    filename = Datei[n]
    #print (filename)
    plt.plot(filename[:,1],filename[:,2],'bo')
    plt.title('Plot des Durchlaufes der Datei%d'%n)
       
    plt.xlabel("Radius r")
    plt.ylabel("Energie")
    n=n+1
    plt.savefig('zusammenPlot der Datei%d.png'%n, dpi=fig.dpi)
    plt.show()"""
    
   
    
    
