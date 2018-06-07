# -*- coding: utf-8 -*- #### für dummis -;-
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
        'CCSD CO 30.txt',
        'CCSD CO 60.txt',
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
    
    
Diss=[]   
####das nennt man eine Funktion definieren 
def plotandfit(Datei1,Datei2,Überschrift,Speicher,R0=1,i=0,xaxes=0,yaxes=0):
    
    #####übergebe Daten an x und y und entwickel y um null 
    y1=np.array(Datei1[:,2])-min(Datei1[:,2])
    y2=np.array(Datei2[:,2])-min(Datei1[:,2])
    x1=np.array(Datei1[:,1])
    x2=np.array(Datei2[:,1])
    
    
    xdata2=np.append(x1,x2)
    ydata2=np.append(y1,y2)
    print (len(xdata2))
    print (len(ydata2))
    
    
    ##################### ab hier wird gefittet  
    t=np.linspace(0.1,7)
    
    tstart = [1.e+3, 1, R0, 0]
    def morse(x, q, m, u , v):
        return (q * (np.exp(-2*m*(x-u))-2*np.exp(-m*(x-u))) + v)
    
    popt, pcov = curve_fit(morse, xdata2, ydata2, p0 = tstart,  maxfev=40000000)
    
    yfit = morse(t,popt[0], popt[1], popt[2], popt[3])
    
    ##### Wenn gewünscht kann hier ausgegeben werden
    #plt.plot(t, yfit)  
           
    #print ('Dissoziationsenergie:',popt[0])
    #print ('Beta:',popt[1])
    #print ('Gleichgewichtsabstand:',popt[2])
    
    
        
    #### R_0 suchen und Dissoziationsenergie 
    mi=min(Datei1[:,2])
    indices = [i for i, x in enumerate(Datei1[:,2]) if x == mi]
    a=Datei1[indices,1]    
    Disso=[max(Datei2[:,2])-min(Datei1[:,2]),str(Überschrift),str(a[0])]
    
    Diss.append(Disso) #### Array für den Save nachher von R0 und Diss.k.
    
    
    
    ####################### ab hier wird der plot allgemein bearbeitet
               
    
    plt.title(Überschrift)   ##für jeden Plot allein ohne subplot
    plt.xlabel("Abstand r in $\AA$")
    plt.ylabel("Energie in Hartree")
    
    ####defaul ist null, wenn gewünscht zum abstellen der Achsen; hilfreich bei subplotting
    frame1 = plt.gca()
    if xaxes == 1: frame1.axes.get_xaxis().set_visible(False)
    if yaxes == 1: frame1.axes.get_yaxis().set_visible(False)
    
    
    
    
    
    #################### wenn gewünscht eine Nullline
    nullline=np.zeros(len(t))   
    plt.plot(t,nullline,'-.')
    
    
    ####steht jetzt immer drüber also HAUPTüberschrift
    plt.suptitle('Potentialkurven des Grundzustandes') 
    
    ####plotte!
    plt.plot(xdata2, ydata2,".", label=Überschrift+' R_0 = '+str(a[0])+' $\AA$')
    plt.legend(loc='lower right')  
    
    ##### jeder Plot sollte für sich gespeichert werden. funktioniet aber nicht so wie gewünscht
    speichername= str(Speicher)    
    plt.savefig('%s.pdf'%(speichername))   
    plt.show()
    
    
    
    
    

#### mach aus Liste Array und vertausche Zeile mit Spalte
def prepare_array(array, yi=2):
        
    a=[[array[i,0],array[i,1],array[i,yi]] for i in range(len(array))]

    a=np.array(a)
    return a





################## ab hier wird die Grafik für B3LYP definiert.

#plt.subplot(2,1,1)
plotandfit(data[0],data[1],Überschrift[0],Speicher[0],1,0,0,0)

#plt.subplot(2,1,2)
plotandfit(data[2],data[3],Überschrift[1],Speicher[1],1,0,0,0)

#plt.title('von HCl und CO') 
#plt.ylim(-0.1,0.6)
#plt.savefig('b3lypzusammen.pdf')
#plt.show()



#################### ab hier HCl CCSD
for i in range(2,5):
    a= prepare_array(data[5],i)
    b= prepare_array(data[4],i)
    plotandfit(b,a,Überschrift[i],Speicher[i],1,i,0,0)

#plt.ylim(-0.2,0.4)
#plt.title('HCl mit Methode CCSD')
#plt.savefig('HCl_CCSD.pdf')     
#plt.show()
        
        
        
###################### ab hier CO CCSD        
for i in range(2,5):   
    a= prepare_array(data[7],i)
    b= prepare_array(data[6],i)
    plotandfit(a,b,Überschrift[i+3],Speicher[i+3],1,i,0,0)
    
    
   
plt.ylim(-0.2,0.7)
plt.title('CO mit Methode CCSD')    
plt.savefig('CO_CCSD.pdf')
plt.show()


##### und noch der Save von diss.k. und R_0
file = open("Disskonst.txt","w")
for item in Diss:
    file.write("%s\n" % item)
file.close()
