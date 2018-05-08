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
        'CCSD CO 30.txt',
        'CCSD CO 60.txt',
        ]

def getData(name):
    
    return np.genfromtxt(str(name), dtype=float, comments='#')
                         
data = [getData(i) for i in names]

Überschrift=np.array([                                                          #n
        ' \n von CO mittels B3LYP',            #0
        ' \n von HCl mittels B3LYP',           #1
        ' HCl mittels CSSD(SCF)',              #2
        ' HCl mittels CSSD (MP2)',             #3
        ' HCl mittels CSSD (MP3)',              #4
         'CO mittels CSSD(SCF)',              #5
        ' CO mittels CSSD (MP2)',             #6
        ' CO mittels CSSD (MP3)', 
        ])

Speicher=np.array([                 
         'CO_b3lyp',                                                    #0
        'HCl_b3lyp',                                                     #1
        'HCl_CSSD(SCF)',                                                    #2
        'HCl_CSSD(MP2)',                                                   #3
        'HCl_CSSD(MP3)',                                                   #4
        'CO_CSSD(SCF)',                                                    #5
        'CO_CSSD(MP2)',                                                   #6
        'CO_CSSD(MP3)',
        ])
    
def plotandfit(Datei1,Datei2,Überschrift,Speicher,R0=1,i=0,xaxes=0):
    
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
    #print ('Beta:',popt[1])
    print ('Gleichgewichtsabstand:',popt[2])
    
    yfit = morse(t,popt[0], popt[1], popt[2], popt[3])
    
    #print popt
    #
    #
    #
    #if i >= 1: plt.subplot(3,1,i-1)
    
    red_patch = mpatches.Patch(color='red', label='Dissoziationsenergie:'+str(popt[0]))
    blue_patch = mpatches.Patch(color='blue', label='Beta:'+str(popt[1]))
    green_patch = mpatches.Patch(color='green', label='Gleichgewichtsabstand:'+str(popt[2]))
    
    #plt.legend(handles=[red_patch,blue_patch,green_patch],prop={'size': 6})#,bbox_to_anchor=(1.05, 1), loc=2 ,borderaxespad=0.)
    if i == 2 or i == 0:plt.title(Überschrift)   
    if i >= 1 or i== 0: plt.xlabel("Abstand r in $\AA$")
    if i == 2 or i == 0:plt.ylabel("Energie in Joule")
    plt.plot(xdata2, ydata2,"x")
    #plt.plot(t, yfit)
    nullline=np.zeros(len(t))
    #plt.plot(t,nullline)
    plt.ylim(-0.05,max(y1)+0.2)
    if i == 0: plt.ylim(-0.05,0.6)
    if i == 0: plt.suptitle('Potentialkurve des Grundzustandes')
    speichername= str(Speicher)
    frame1 = plt.gca()
    plt.savefig('%s.pdf'%(speichername))
    #plt.axis('off')
    #plt.xticks(off)
    ##ax.set_xticks([])
    if i >= 1 and i < 4: frame1.axes.get_xaxis().set_visible(False)
    frame1 = plt.gca()
    frame1 = plt.gca()
    if xaxes == 1: frame1.axes.get_xaxis().set_visible(False)
    #plt.xlabel()
    
    
    
    


def prepare_array(array, yi=2):
        
    a=[[array[i,0],array[i,1],array[i,yi]] for i in range(len(array))]

    a=np.array(a)
    return a
a= prepare_array(data[7],2)
#print (a)
#print (data[4])
#plt.subplot(2,1,1)
plotandfit(data[0],data[1],Überschrift[0],Speicher[0],1,0,1)

#plt.subplot(2,1,2)
plotandfit(data[2],data[3],Überschrift[1],Speicher[1],1,0,0)



plt.savefig('b3lypzusammen.pdf')
plt.show()

for i in range(2,5):
    a= prepare_array(data[5],i)
    b= prepare_array(data[4],i)
    plotandfit(b,a,Überschrift[3],Speicher[i],1,i)
    
plt.show()
        #print (a)
for i in range(2,5):
    
    a= prepare_array(data[7],i)
    b= prepare_array(data[6],i)
    plotandfit(a,b,Überschrift[6],Speicher[i],1,i)
    #print (b-min(b[:,2]))
    
plt.show()



