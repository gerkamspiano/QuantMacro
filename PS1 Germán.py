#Germán Sánchez Arce
#In colaboration with Maria Gonzalez & Joan Alegre

#%%Taylor series expansion


#Import packages
from sympy import symbols
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylim
x=symbols('x')

#definition of the factorial function
def factorial(n):
    if n <= 0:
        return 1
    else:
        return n * factorial(n - 1)

##Exercise 1
##Taylor series around a=1
        
a=1 
f=x**0.321  #function 
taylor1=f.subs(x,a)+f.diff(x).subs(x,a)*(x-a)  #using the formula of the first order approximation
print("the first order approximation solution around x=1 is",taylor1)

def taylor(f,a,n):      #general formula of the Taylor expansion
    k = 0           
    p = 0
    while k <= n:
        p = p + (f.diff(x,k).subs(x,a))/(factorial(k))*(x-a)**k
        k += 1
    return p

print("using the general formula we have the same result:", taylor(f,1,1))

#Repeat with different orders
print("2nd order approximation is:",taylor(f,1,2))   #2nd order approximation around a=1
print("")
print("5th order approximation is:", taylor(f,1,5))    #2nd order approximation around a=1
print("")
print("20th order approximation is:", taylor(f,1,20))      #2nd order approximation around a=1
print("")

x=np.linspace(0,4,50)       #points in the x axis to plot the function in the range (0,4)

#we define the results of the Taylor series approximations

y0=x**0.321
y1 = 0.321*x + 0.679
y2 = 0.321*x - 0.1089795*(x - 1)**2 + 0.679
y5 = 0.321*x + 0.0300570779907967*(x - 1)**5 - 0.040849521596625*(x - 1)**4 + 0.0609921935*(x - 1)**3 - 0.1089795*(x - 1)**2 + 0.679
y20 = 0.321*x - 0.00465389246518441*(x - 1)**20 + 0.00498302100239243*(x - 1)**19 - 0.00535535941204005*(x - 1)**18 + 0.00577951132662155*(x - 1)**17 - 0.00626645146709397*(x - 1)**16 + 0.00683038514023459*(x - 1)**15 - 0.00749000490558658*(x - 1)**14 + 0.0082703737422677*(x - 1)**13 - 0.00920582743809231*(x - 1)**12 + 0.0103445949299661*(x - 1)**11 - 0.0117564360191783*(x - 1)**10 + 0.0135458417089277*(x - 1)**9 - 0.0158761004532294*(x - 1)**8 + 0.0190161406836106*(x - 1)**7 - 0.0234395113198229*(x - 1)**6 + 0.0300570779907967*(x - 1)**5 - 0.040849521596625*(x - 1)**4 + 0.0609921935*(x - 1)**3 - 0.1089795*(x - 1)**2 + 0.679

bottom,top = ylim() 

#define the range of the y axis (-2,7)
ylim(top=7)             
ylim(bottom=-2)

#plot the results
plt.title('Taylor approximations',size=20)
plt.plot(x,y0, label='T0')
plt.plot(x,y1, label='T1')
plt.plot(x,y2, label='T2')
plt.plot(x, y5, label='T5')
plt.plot(x,y20, label='T20')
plt.legend(loc='upper left')
plt.xlabel('x',size=15)
plt.ylabel('f(x)',size=15)
plt.show()

#%% Exercise 2

##Taylor series around a=1
a=2
x=symbols('x')
f=(2*x)/2 #function


def taylor(f,a,n): #Taylor function
    k = 0
    p = 0
    while k <= n:
        p = p + (f.diff(x,k).subs(x,a))/(factorial(k))*(x-a)**k
        k += 1
    return p

#Print Taylor series

print("")
print("")
print("2nd exercise")
print("")
print("")
print("2nd order Taylor approximation is:", taylor(f,2,2))
print("")
print("5th order Taylor approximation is:", taylor(f,2,5))
print("")
print("20th order Taylor approximation is:", taylor(f,2,20))
print("")

x=np.linspace(-2,6,50)

y0=(x+abs(x))/2 #original function
y1 = x
y2 = x
y5 = x
y20 = x


#define the range of the y axis (-2,7)
bottom,top = ylim()
ylim(top=7)
ylim(bottom=-2)

#plot of the Taylor series approximations
plt.title('Taylor series approximation of a Rump function',size=20)
plt.plot(x,y0, label='T0')
plt.plot(x,y1, label='T1')
plt.plot(x,y2, label='T2')
plt.plot(x, y5, label='T5')
plt.plot(x,y20, label='T20')
plt.legend(loc='upper left')
plt.xlabel('x',size=15)
plt.ylabel('f(x)',size=15)
plt.show()

#%%Exercise 3

import matplotlib.pyplot as plt
import numpy as np



#""function 1

e=2.71828182 #Euler number
x = np.linspace(-1, 1, num=20, endpoint=True) #I did in this range because otherwise, since there is a discontinuity in 0, the plot would not show anything in this case
xf = np.linspace(-1, 1, num = 20, endpoint=True)
y=e**(1/x)
y2=e**(1/xf)

cheb3=np.polyfit(x,y,3)    #Monomial approximation of degree 3
val3=np.polyval(cheb3,x) 
cheb5=np.polyfit(x,y,5)    #Monomial approximation of degree 5
val5=np.polyval(cheb5,x)
cheb10=np.polyfit(x,y,10)  ##Monomial approximation of degree 10
val10=np.polyval(cheb10,x)

plt.plot(x,val3,':',label='T3')
plt.plot(x,val5,':',label='T5')
plt.plot(x,val10,':',label='T10')
plt.plot(xf,y2,label='T0')
plt.plot(xf,y2,'o')
plt.title('Approximation by interpolation',size=15)
plt.legend(loc='upper left')
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.show()



#errors of interpolation approximation
x = np.linspace(-1, 1, num = 20, endpoint=True)

def y(x):
    return e**(1/x)

y1 = y(x)

error3 = abs(y1-val3) #error of the 3rd order approximation
error5 = abs(y1-val5) #error of the 5th order approximation
error10 = abs(y1-val10) #error of the 10th order approximation

plt.title('Errors of approximations by interpolation',size=15)
plt.plot(x,error3,'-',label='error T3')
plt.plot(x,error5,':',label='error T5')
plt.plot(x,error10,'--',label='error T10')
plt.legend(loc='upper left')
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.show()

#Monomial approximation with Chebyshev's nodes

def f(x):
    return e**(1/x)

vector=np.linspace(-1,1,24)
ch=np.polynomial.chebyshev.chebroots(vector)

y=f(ch)    
cheb3=np.polyfit(ch,y,3)    #Monomial approximation of degree 3
val3=np.polyval(cheb3,ch) 
cheb5=np.polyfit(ch,y,5)    #Monomial approximation of degree 5
val5=np.polyval(cheb5,ch)
cheb10=np.polyfit(ch,y,10)  ##Monomial approximation of degree 10
val10=np.polyval(cheb10,ch)


plt.title('Monomial approximations with Cheb. nodes',size=15)
plt.plot(ch,val3,'--',label='T3')
plt.plot(ch,val5,'--',label='T5')
plt.plot(ch,val10,':',label='T10')
plt.plot(ch,f(ch),label='T0')
plt.plot(ch,f(ch),'o')
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.legend(loc='upper left')
plt.show()

#Errors of monomial approximation with Cheb. nodes 

y1 = f(ch)

error3 = abs(y1-val3)
error5 = abs(y1-val5)
error10 = abs(y1-val10)

plt.plot(ch, error3,'-', label = 'Error of order 3')
plt.plot(ch, error5,'--', label = 'Error of order 5')
plt.plot(ch, error10, ':', label = 'Error of order 10')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Monomial approx. errors with Cheb. nodes', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()


#Chebyshev polynomial approximation with Chebyshev's nodes

def f(x):
    return e**(1/x)

vector=np.linspace(-1,1,24)
ch=np.polynomial.chebyshev.chebroots(vector)

y=f(ch)    
cheb3=np.polynomial.chebyshev.chebfit(ch,y,3)    #Chebyshev approximation of degree 3
val3=np.polynomial.chebyshev.chebval(ch,cheb3) 
cheb5=np.polynomial.chebyshev.chebfit(ch,y,5)    #Chebyshev approximation of degree 5
val5=np.polynomial.chebyshev.chebval(ch,cheb5)
cheb10=np.polynomial.chebyshev.chebfit(ch,y,10)  ##Chebyshev approximation of degree 10
val10=np.polynomial.chebyshev.chebval(ch,cheb10)


plt.title('Cheb. poly. approximations with Cheb. nodes',size=15)
plt.plot(ch,val3,'--',label='T3')
plt.plot(ch,val5,'--',label='T5')
plt.plot(ch,val10,':',label='T10')
plt.plot(ch,f(ch),label='T0')
plt.plot(ch,f(ch),'o')
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.legend(loc='upper left')
plt.show()

#Errors of Chebyshev's polynomial approximation with Cheb. nodes 

y1 = f(ch)

error3 = abs(y1-val3)
error5 = abs(y1-val5)
error10 = abs(y1-val10)

plt.plot(ch, error3,'-', label = 'Error of order 3')
plt.plot(ch, error5,'--', label = 'Error of order 5')
plt.plot(ch, error10, ':', label = 'Error of order 10')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Cheb. poly. approx. errors with Cheb. nodes', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

#%%
#""function 2

def f(x):
    return 1/(1+25*x**2)

x = np.linspace(-1, 1, num=40, endpoint=True) #I did in this range because otherwise, since there is a discontinuity in 0, the plot would not show anything in this case
y=f(x)

cheb3=np.polyfit(x,y,3)    #Monomial approximation of degree 3
val3=np.polyval(cheb3,x) 
cheb5=np.polyfit(x,y,5)    #Monomial approximation of degree 5
val5=np.polyval(cheb5,x)
cheb10=np.polyfit(x,y,10)  ##Monomial approximation of degree 10
val10=np.polyval(cheb10,x)

plt.plot(x,val3,':',label='T3')
plt.plot(x,val5,':',label='T5')
plt.plot(x,val10,':',label='T10')
plt.plot(x,y,label='T0')
plt.plot(x,y,'o')
plt.title('Approximation by interpolation',size=15)
plt.legend(loc='upper left')
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.show()



#errors of interpolation approximation
x = np.linspace(-1, 1, num = 40, endpoint=True)

def y(x):
    return 1/(1+25*x**2)

y1 = y(x)

error3 = abs(y1-val3) #error of the 3rd order approximation
error5 = abs(y1-val5) #error of the 5th order approximation
error10 = abs(y1-val10) #error of the 10th order approximation

plt.title('Errors of approximations by interpolation',size=15)
plt.plot(x,error3,'-',label='error T3')
plt.plot(x,error5,':',label='error T5')
plt.plot(x,error10,'--',label='error T10')
plt.legend(loc='upper left')
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.show()

#Monomial approximation with Chebyshev's nodes

def f(x):
    return 1/(1+25*x**2)

vector=np.linspace(-1,1,40)
vectorO=np.linspace(-1,1,40)
ch=np.polynomial.chebyshev.chebroots(vector)
chO=np.polynomial.chebyshev.chebroots(vectorO)

y=f(ch)    
cheb3=np.polyfit(ch,y,3)    #Monomial approximation of degree 3
val3=np.polyval(cheb3,ch) 
cheb5=np.polyfit(ch,y,5)    #Monomial approximation of degree 5
val5=np.polyval(cheb5,ch)
cheb10=np.polyfit(ch,y,10)  ##Monomial approximation of degree 10
val10=np.polyval(cheb10,ch)


plt.title('Monomial approximations with Cheb. nodes',size=15)
plt.plot(ch,val3,'--',label='T3')
plt.plot(ch,val5,'--',label='T5')
plt.plot(ch,val10,':',label='T10')
plt.plot(ch,f(ch),label='T0')
plt.plot(chO,f(chO),'o')
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.legend(loc='upper left')
plt.show()

#Errors of monomial approximation with Cheb. nodes 

y1 = f(ch)

error3 = abs(y1-val3)
error5 = abs(y1-val5)
error10 = abs(y1-val10)

plt.plot(ch, error3,'-', label = 'Error of order 3')
plt.plot(ch, error5,'--', label = 'Error of order 5')
plt.plot(ch, error10, ':', label = 'Error of order 10')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Monomial approx. errors with Cheb. nodes', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()


#Chebyshev polynomial approximation with Chebyshev's nodes

def f(x):
    return 1/(1+25*x**2)
x = np.linspace(-1, 1, num = 40, endpoint=True)
vector=np.linspace(-1,1,40)
ch=np.polynomial.chebyshev.chebroots(vector)

y=f(ch)    
cheb3=np.polynomial.chebyshev.chebfit(ch,y,3)    #Chebyshev approximation of degree 3
val3=np.polynomial.chebyshev.chebval(ch,cheb3) 
cheb5=np.polynomial.chebyshev.chebfit(ch,y,5)    #Chebyshev approximation of degree 5
val5=np.polynomial.chebyshev.chebval(ch,cheb5)
cheb10=np.polynomial.chebyshev.chebfit(ch,y,10)  ##Chebyshev approximation of degree 10
val10=np.polynomial.chebyshev.chebval(ch,cheb10)


plt.title('Cheb. poly. approximations with Cheb. nodes',size=15)
plt.plot(ch,val3,'--',label='T3')
plt.plot(ch,val5,'--',label='T5')
plt.plot(ch,val10,':',label='T10')
plt.plot(ch,f(ch),label='T0')
plt.plot(ch,f(ch),'o')
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.legend(loc='upper left')
plt.show()

#Errors of Chebyshev's polynomial approximation with Cheb. nodes 

y1 = f(ch)

error3 = abs(y1-val3)
error5 = abs(y1-val5)
error10 = abs(y1-val10)

plt.plot(ch, error3,'-', label = 'Error of order 3')
plt.plot(ch, error5,'--', label = 'Error of order 5')
plt.plot(ch, error10, ':', label = 'Error of order 10')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Cheb. poly. approx. errors with Cheb. nodes', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

#%%
#""function 3

def f(x):
    return (x+abs(x))/2

x = np.linspace(-1, 1, num=40, endpoint=True) #I did in this range because otherwise, since there is a discontinuity in 0, the plot would not show anything in this case
y=f(x)

cheb3=np.polyfit(x,y,3)    #Monomial approximation of degree 3
val3=np.polyval(cheb3,x) 
cheb5=np.polyfit(x,y,5)    #Monomial approximation of degree 5
val5=np.polyval(cheb5,x)
cheb10=np.polyfit(x,y,10)  ##Monomial approximation of degree 10
val10=np.polyval(cheb10,x)

plt.plot(x,val3,':',label='T3')
plt.plot(x,val5,':',label='T5')
plt.plot(x,val10,':',label='T10')
plt.plot(x,y,label='T0')
plt.plot(x,y,'o')
plt.title('Approximation by interpolation',size=15)
plt.legend(loc='upper left')
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.show()



#errors of interpolation approximation
x = np.linspace(-1, 1, num = 40, endpoint=True)

def y(x):
    return  (x+abs(x))/2

y1 = y(x)

error3 = abs(y1-val3) #error of the 3rd order approximation
error5 = abs(y1-val5) #error of the 5th order approximation
error10 = abs(y1-val10) #error of the 10th order approximation

plt.title('Errors of approximations by interpolation',size=15)
plt.plot(x,error3,'-',label='error T3')
plt.plot(x,error5,':',label='error T5')
plt.plot(x,error10,'--',label='error T10')
plt.legend(loc='upper left')
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.show()

#Monomial approximation with Chebyshev's nodes

def f(x):
    return  (x+abs(x))/2
vector=np.linspace(-1,1,30)
ch=np.polynomial.chebyshev.chebroots(vector)

y=f(ch)    
cheb3=np.polyfit(ch,y,3)    #Monomial approximation of degree 3
val3=np.polyval(cheb3,ch) 
cheb5=np.polyfit(ch,y,5)    #Monomial approximation of degree 5
val5=np.polyval(cheb5,ch)
cheb10=np.polyfit(ch,y,10)  ##Monomial approximation of degree 10
val10=np.polyval(cheb10,ch)


plt.title('Monomial approximations with Cheb. nodes',size=15)
plt.plot(ch,val3,'--',label='T3')
plt.plot(ch,val5,'--',label='T5')
plt.plot(ch,val10,':',label='T10')
plt.plot(ch,f(ch),'o')
plt.plot(ch,f(ch),label='T0')
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.legend(loc='upper left')
plt.show()

#Errors of monomial approximation with Cheb. nodes 

y1 = f(ch)

error3 = abs(y1-val3)
error5 = abs(y1-val5)
error10 = abs(y1-val10)

plt.plot(ch, error3,'-', label = 'Error of order 3')
plt.plot(ch, error5,'--', label = 'Error of order 5')
plt.plot(ch, error10, ':', label = 'Error of order 10')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Monomial approx. errors with Cheb. nodes', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()


#Chebyshev polynomial approximation with Chebyshev's nodes

def f(x):
    return  (x+abs(x))/2


vector=np.linspace(-1,1,30)
ch=np.polynomial.chebyshev.chebroots(vector)
chO=np.polynomial.chebyshev.chebroots(vectorO)

y=f(ch)    
cheb3=np.polynomial.chebyshev.chebfit(ch,y,3)    #Chebyshev approximation of degree 3
val3=np.polynomial.chebyshev.chebval(ch,cheb3) 
cheb5=np.polynomial.chebyshev.chebfit(ch,y,5)    #Chebyshev approximation of degree 5
val5=np.polynomial.chebyshev.chebval(ch,cheb5)
cheb10=np.polynomial.chebyshev.chebfit(ch,y,10)  ##Chebyshev approximation of degree 10
val10=np.polynomial.chebyshev.chebval(ch,cheb10)


plt.title('Cheb. poly. approximations with Cheb. nodes',size=15)
plt.plot(ch,val3,'--',label='T3')
plt.plot(ch,val5,'--',label='T5')
plt.plot(ch,val10,':',label='T10')
plt.plot(ch,f(ch),label='T0')
plt.plot(ch,f(ch),'o')
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.legend(loc='upper left')
plt.show()

#Errors of Chebyshev's polynomial approximation with Cheb. nodes 

y1 = f(ch)

error3 = abs(y1-val3)
error5 = abs(y1-val5)
error10 = abs(y1-val10)

plt.plot(ch, error3,'-', label = 'Error of order 3')
plt.plot(ch, error5,'--', label = 'Error of order 5')
plt.plot(ch, error10, ':', label = 'Error of order 10')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Cheb. poly. approx. errors with Cheb. nodes', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()


#%%Exercise 4

import numpy as np
import matplotlib.pyplot as plt

#x=symbols('x')
e=2.7182
a=1.0
p2=1/100
p1=1/0.2

def f(x):
    return (e**(-a*x))/(p1+p2*e**(-a*x))

x = np.linspace(0, 10, num = 20, endpoint=True)
y=f(x)    
cheb3=np.polynomial.chebyshev.chebfit(x,y,3)    #Chebyshev approximation of degree 3
val3=np.polynomial.chebyshev.chebval(x,cheb3) 
cheb5=np.polynomial.chebyshev.chebfit(x,y,5)    #Chebyshev approximation of degree 5
val5=np.polynomial.chebyshev.chebval(x,cheb5)
cheb10=np.polynomial.chebyshev.chebfit(x,y,10)  ##Chebyshev approximation of degree 10
val10=np.polynomial.chebyshev.chebval(x,cheb10)


plt.title('Cheb. poly. approximations, p1=1/0.25',size=15)
plt.plot(x,val3,'--',label='T3')
plt.plot(x,val5,'--',label='T5')
plt.plot(x,val10,'--',label='T10')
plt.plot(x,f(x),label='T0')
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.legend(loc='upper right')
plt.show()


#report errors

error1 = abs(val3-y) #error of the 3srd order approximation
error3 = abs(val5-y) #error of the 5th order approximation
error5 = abs(val10-y) #error of the 10th order approximation

plt.title('Errors of Chebyshevs approximations, p1=1/0.25',size=15)
plt.plot(x,error1,':',label='error T3')
plt.plot(x,error3,'-',label='error T5')
plt.plot(x,error5,'--',label='error T10')
plt.legend(loc='upper right')
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.show()


#%% Now with p1=1/0.25

x = np.arange(0, 20, 0.5)
e=2.7182
a=1.0
p2=1/100
p1=1/0.25
p=(e**(-a*x))/(p1+p2*e**(-a*x))

def f(x):
    return (e**(-a*x))/(p1+p2*e**(-a*x))

x = np.linspace(0, 10, num = 20, endpoint=True)
y=f(x)    
cheb3=np.polynomial.chebyshev.chebfit(x,y,3)    #Chebyshev approximation of degree 3
val3_=np.polynomial.chebyshev.chebval(x,cheb3) 
cheb5=np.polynomial.chebyshev.chebfit(x,y,5)    #Chebyshev approximation of degree 5
val5_=np.polynomial.chebyshev.chebval(x,cheb5)
cheb10=np.polynomial.chebyshev.chebfit(x,y,10)  ##Chebyshev approximation of degree 10
val10_=np.polynomial.chebyshev.chebval(x,cheb10)


plt.title('Cheb. poly. approximations p1=1/0.2',size=15)
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.plot(x,val3_,'--')
plt.plot(x,val5_,'.')
plt.plot(x,val10_,':')
plt.plot(x,f(x))
plt.show()


#report errors
x = np.linspace(0, 10, num = 20, endpoint=True)

error1_2 = abs(val3_-y) #error of the 3srd order approximation
error3_2 = abs(val5_-y) #error of the 5th order approximation
error5_2 = abs(val10_-y) #error of the 10th order approximation

plt.title('Errors of Chebyshevs approximations p1=1/0.2',size=15)
plt.plot(x,error1_2,':',label='error T3')
plt.plot(x,error3_2,'-',label='error T5')
plt.plot(x,error5_2,'--',label='error T10')
plt.legend(loc='upper right')
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.show()


#%%Combining p1=1/0.25 and p1=1/0.20

plt.title('Cheb. poly. approximations both cases',size=15)
plt.plot(x,val3,':',label='T3')
plt.plot(x,val5,':',label='T5')
plt.plot(x,val10,':',label='T10')
plt.plot(x,val3_,'--',label='T3_2')
plt.plot(x,val5_,'--',label='T5_2')
plt.plot(x,val10_,'--',label='T10_2')
plt.plot(x,f(x),label='T0')
plt.legend(loc='upper right')
plt.xlabel('x',size=10)
plt.ylabel('f(x)',size=10)
plt.show()


plt.title('Errors of Chebyshevs approximations, both cases',size=15)
plt.plot(x,error1,':',label='error T3')
plt.plot(x,error3,':',label='error T5')
plt.plot(x,error5,':',label='error T10')
plt.plot(x,error1_2,'--',label='error T3_2')
plt.plot(x,error3_2,'--',label='error T5_2')
plt.plot(x,error5_2,'--',label='error T10_2')
plt.legend(loc='upper right')
plt.xlabel('x',size=10)
plt.ylabel('val-y',size=10)
plt.show()


#%%Question 2

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols
from sympy import solve

h=symbols('h')
k=symbols('k')
a=.5
s=.25

y=((1-a)*k**((s-1)/s)+a*h**((s-1)/s))**(s/(s-1))

def f(k,h):
    return ((1-a)*k**((s-1)/s)+a*h**((s-1)/s))**(s/(s-1))
max= f(10,10)

per5=max*0.05 #percentil 5
hs=solve(y-0.5,h)



k=np.linspace(0,10,num=100,endpoint=True)
plt.title('Output percentiles',size=20)
plt.plot(k,((0.5**3)/2-k**3)**(1/3),label='P5') #per 5
plt.plot(k,((1**3)/2-k**3)**(1/3),label='P10') #per10
plt.plot(k,((2.5**3)/2-k**3)**(1/3),label='P25') #per25
plt.plot(k,((5**3)/2-k**3)**(1/3),label='P50') #per50
plt.plot(k,((7.5**3)/2-k**3)**(1/3),label='P75') #per75
plt.plot(k,((9**3)/2-k**3)**(1/3),label='P90') #per90
plt.plot(k,((9.5**3)/2-k**3)**(1/3),label='P95') #per95
plt.legend(loc='upper right')
plt.xlabel('h',size=15)
plt.ylabel('k',size=15)
plt.show()




















