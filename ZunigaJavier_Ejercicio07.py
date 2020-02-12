import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




def probabi(Ynu,Yver,sig):
    difi= np.subtract(x,mu)
    difis=np.square(difi)
    difido= np.sum(difis)
    pro=np.exp(-difido/(2*sig**(2)))*(1/(sig*np.sqrt(2*np.pi)))**(-len(x))
    return pro
 
#Evaluo mis betas*X
def GENYn(x,beta):
    Ysup= np.zeros(len(x))
    New= beta0[:,1:]
    for i in range(len(x)):
        Ysup[i]=beta[i,0]+np.dot(New[i],x[i])
    return Ysup

# Leemos los datos y se lo asignamos a X y Y
data = np.loadtxt("notas_andes.dat", skiprows=1)
Y = data[:,-1]
X= data[:,0:-1]
beta0=np.random.random((len(Y),len(X[0])+1))


def revis(Ya,beta,X,pasos):
    betahisto=[beta]
    for j in range (pasos):
        betaan =betahisto[-1]
        for i in range (len(Ya)):
            Ynue=GENYn(X,betaan[i])
      
    return betahisto
print(revis(Y,beta0,X,10000))


            
#historico=revis(Y,GENYn(X,beta0),beta0,X,5)

#print(historico)



                
 