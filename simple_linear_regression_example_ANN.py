from matplotlib import pyplot as plt
import numpy as np

from utils import *#my function definitions
"""halilibo mut"""

#my sample dataset
price=[4, 5, 6, 5, 9, 12, 13, 16, 14, 17, 16, 19, 22, 22, 21, 18]# Y=f(x)
sq_ft=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]#X

#---------plotting house price changes by square_feet---------
fig1=plt.figure("simple linear regression brute force-halilibo mut")
plt.plot(sq_ft,price,"go",label="prices by sqft")

plt.ylabel("house price")
plt.xlabel("square feet")
#-------------------------------------------------------------

brute_f_list=[]#estimades functions results
funcs=[]#to print functions as strings
err_l=[]#error rate of function values


def brute_force():
    """produces the linear solution models by given ranges and precision"""
    global brute_f_list
    global funcs
    
    ran=3.2#range of brute force
    precision=0.1#choose bigger since its brute forcing
    round_coeff=len( str(precision).split(".")[1] )
    
    for a in np.arange(0, ran, precision):
        temp=[]
    
        for b in np.arange(0.1, ran, precision):
            for x in sq_ft:
                out=f(x=x, a=a, b=b)
                temp.append(out)
            funcs.append( f"{a:.{round_coeff}f}x+{b:.{round_coeff}f}" )
            brute_f_list.append(temp)
            temp=[]

def calc_errors():
    global brute_f_list
    global err_l
    
    for a in brute_f_list:
        #ei=std_dev_mean(a, mean(price))
        ei=std_dev(price,a)
        err_l.append(ei)

brute_force()
calc_errors()
indexes=cis(err_l,5)#find the #(5) of indexes that have least error rates




#---approved solution by field of economics---
x=sq_ft
y=price
l=len(x)

m_x=mean(x)*l
m_y=mean(y)*l
m_xy=mxy(x,y)
mx2=m2(x)
my2=m2(y)

a=coeff_x(m_x, m_y, m_xy, mx2, l)
b=const_x(m_x,m_y,a,l)

"""print(f"calc estimated Y=f(x):{a}x+{b}")"""
fx=[]#formula results of field of economics

for out in x:
    fx.append( f(out,a,b) )
plt.plot(x,fx,"o-",label=f"{a:.4f}x+{b:.4f} (economics formula)")

#---approved solution by field of economics---





#-----plot the results that have minimal error rates-----
for _ in indexes:
    plt.plot(sq_ft, brute_f_list[_],"--",label=f"y={ funcs[_] } err:{err_l[_]:.4f}")
#--------------------------------------------------------

"""#!!! VERY SLOW TO DRAW THIS !!!
#plot ALL brute force function results
fig2=plt.figure("all functions draw")
for a,b,c in zip(brute_f_list,funcs,err_l):
    plt.plot(sq_ft,a,"--",label=f"y={ b } err:{c:.3f}")

fig2.legend()#prints labels
"""

fig1.legend(loc=2)
plt.show()
