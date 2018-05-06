# -*- coding: utf-8 -*-
"""
Created on Sun May  6 17:42:22 2018

@author: David
"""

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
names = [
        'CO.txt',
        'CO30.txt',
        'HCL.txt',
        'HCL30.txt',
        'hclco.txt',
        'hclco30.txt',
        ]

def getData(name):
    
    return np.genfromtxt(str(name), dtype=float, comments='#')
                         
data = [getData(i) for i in names]

Überschrift=np.array([
        'Potentialkurve des Grundzustandes \n von CO mittels B3LYP',
        'Potentialkurve des Grundzustandes \n von HCl mittels B3LYP',
        'Potentialkurve des Grundzustandes \n  mittels CSSD(SCF)',
        'Potentialkurve des Grundzustandes \n  mittels CSSD (MP2)',
        'Potentialkurve des Grundzustandes \n mittels CSSD (MP3)',
        'keine'
        ])

Speicher=np.array([
         'CO mittels B3Lyp',
        'Cl mittels B3LYP',
        'mittels CSSD(SCF)',
        'mittels CSSD (MP2)',
        'mittels CSSD (MP3)',
        ])
    
def plotandfit(Datei1,Datei2,Überschrift,Speicher,R0=1):
    
    y1=np.array(Datei1[:,2])-min(Datei1[:,2])
    y2=np.array(Datei2[:,2])-min(Datei1[:,2])
    x1=np.array(Datei1[:,1])
    x2=np.array(Datei2[:,1])
    
    
    xdata2=np.append(x1,x2)
    ydata2=np.append(y1,y2)
    
    #print (data[1])    
    t=np.linspace(0.1,7)
    
    tstart = [1.e+3, 1, R0, 0]
    def morse(x, q, m, u , v):
        return (q * (np.exp(-2*m*(x-u))-2*np.exp(-m*(x-u))) + v)
    
    popt, pcov = curve_fit(morse, xdata2, ydata2, p0 = tstart,  maxfev=40000000)
    print ('Dissoziationsenergie:',popt[0]) # [    5.10155662     1.43329962     1.7991549  -1378.53461345]
    print ('Beta:',popt[1])
    print ('Gleichgewichtsabstand:',popt[2])
    
    yfit = morse(t,popt[0], popt[1], popt[2], popt[3])
    
    #print popt
    #
    #
    #
    plt.subplot(111)
    red_patch = mpatches.Patch(color='red', label='Dissoziationsenergie:'+str(popt[0]))
    blue_patch = mpatches.Patch(color='blue', label='Beta:'+str(popt[1]))
    green_patch = mpatches.Patch(color='green', label='Gleichgewichtsabstand:'+str(popt[2]))
    
    plt.legend(handles=[red_patch,blue_patch,green_patch],prop={'size': 6})#,bbox_to_anchor=(1.05, 1), loc=2 ,borderaxespad=0.)
    plt.title(Überschrift)   
    plt.xlabel("Abstand r in $\AA$")
    plt.ylabel("Energie in Joule")
    plt.plot(xdata2, ydata2,"x")
    plt.plot(t, yfit)
    nullline=np.zeros(len(t))
    plt.plot(t,nullline)
    plt.ylim(-0.1,max(y1)+0.2)
    speichername= str(Speicher)
    plt.savefig('%s.png'%(speichername))
    
    plt.show()
    


def prepare_array(array, yi=2):
        
    a=[[array[i,0],array[i,1],array[i,yi]] for i in range(len(array))]

    a=np.array(a)
    return a
a= prepare_array(data[3],2)



plotandfit(data[0],data[1],Überschrift[0],Speicher[0],0.9)
plotandfit(data[2],a,Überschrift[1],Speicher[1],3)


for i in range(2,5):
    a= prepare_array(data[5],i)
    plotandfit(data[4],data[5],Überschrift[i],Speicher[i],1)
    #print (a)



