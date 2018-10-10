
# coding: utf-8

# In[2]:


# Ps3 Germán Sánchez Arce 

#%% Packages
import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import scipy.optimize as sc
import numpy as np
from scipy.optimize import fsolve
from numpy import random
from numpy import *
from scipy.optimize import *
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
#%%Exercice 1. a) and b):
# Let's find Z:
y = 1
ht = (31/100)
b = 0.99 # beta
def zeta(z):
    ht = (31/100)
    k = 4
    z = z
    f = pow(k,0.33)*pow(z*ht,0.67)-1
    return f
zguess = 1
z = sc.fsolve(zeta, zguess)
#Delta (depreciation) is clearly 1/16.
d = 1/16 # depreciation
h2z = pow((2*z*ht),0.67) #new worker in efficient terms.
def SS(k):
    k = k
    return 0.33*pow(k,-0.67)*h2z+1-d-(1/b)
kguess = 1
k = sc.fsolve(SS,kguess)
print('9.6815 is the new stationary capital when 2 times z')
#%%Exercice 1. c))
def EE(k1,k2,k3):
    return pow(k2,0.33)*h2z-k3+(1-d)*k2-b*(0.33*pow(k2,-0.67)*h2z+(1-d))*(pow(k1,0.33)*h2z-k2+(1-d)*k1)
#def transv(k1,k2):
#    return pow(b,500)*(0.33*pow(k1,-0.67)*h2z+(1-d))*k1*pow(pow(k1,0.33)*h2z-k2+(1-d)*k1,-1)
K = 9.68
def transition(z): 
    F = np.zeros(200)
    z = z
    F[0] = EE(4,z[1],z[2])
#    F[499] = transv(z[497],z[498])
    z[199] = 9.68
    F[198] = EE(z[197], z[198], z[199])
    for i in range(1,198):
        F[i] = EE(z[i],z[i+1],z[i+2])
    return F
z = np.ones(200)*4
k = sc.fsolve(transition, z)
k[0] = 4
# I create the domain to plot everything.
kplot = k[0:100]
t = np.linspace(0,100,100)

# I create savings, output and consumption:
yt = pow(kplot,0.33)*h2z
kt2 = k[1:101]
st = kt2-(1-d)*kplot
ct = yt-st

plt.plot(t,kplot, label='capital')
plt.legend()
plt.title('Transition of K from  first S.S to second S.S, first 100 times', size=20)
plt.xlabel('Time')
plt.ylabel('capital')
plt.show()

plt.plot(t,yt, label='Yt output')
plt.plot(t,st, label='st savings')
plt.plot(t,ct, label='ct consumption')
plt.legend(loc='upper right')
plt.title('Transition of the economy', size=20)
plt.xlabel('Time', size = 20)
plt.ylabel('Quantity', size = 20)
plt.show()
#%% Exercice1. d):

y = 1
ht = (31/100)
b = 0.99 # beta
def zeta(z):
    ht = (31/100)
    k = 4
    z = z
    f = pow(k,0.33)*pow(z*ht,0.67)-1
    return f
zguess = 1
z = sc.fsolve(zeta, zguess)

#new hz
hz = pow((z*ht),0.67)

# New euler equation:
def EE2(k1,k2,k3):
    return pow(k2,0.33)*hz-k3+(1-d)*k2-b*(0.33*pow(k2,-0.67)*hz+(1-d))*(pow(k1,0.33)*hz-k2+(1-d)*k1)

k10 = k[9]
# I compute the new Stady stationary:
def SS(k):
    k = k
    return 0.33*pow(k,-0.67)*hz+1-d-(1/b)
kguess = 1
kss = sc.fsolve(SS,kguess)

def transition(z): 
    F = np.zeros(100)
    z = z
    F[0] = EE2(6.801,z[1],z[2])
#    F[499] = transv(z[497],z[498])
    z[99] = 4.84
    F[98] = EE2(z[97], z[98], z[99])
    for i in range(1,98):
        F[i] = EE2(z[i],z[i+1],z[i+2])
    return F
z = np.ones(100)*4
k2 = sc.fsolve(transition, z)
k2[0] = 6.801

#lets plot everything:
kplot = k[0:100]
kfin = np.append(k[0:10],k2[0:90]) 
t = np.linspace(0,100,100)
plt.plot(t,kplot,'--', label='expected transition')
plt.plot(t,kfin, label='actual transition')
plt.axvline(x=10, color='black')
plt.legend()
plt.title('Difference of economy by shock at t=10', size=20)
plt.xlabel('Time', size=20)
plt.ylabel('Capital', size = 20)
plt.show()

