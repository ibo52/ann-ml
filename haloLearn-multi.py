from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from utils import *#my function definitions
"""halilibo mut"""

from simple_linear_regression_example_ANN import *

def multi_f(stepsize):
    """calculates (a2)x^2 + f(x)=ax +b"""
    global price
    global sq_ft
    
    global funcs
    global err_l
    global brute_f_list

    brute_f_list=brute_f_list[-1:]
    
    funcs=funcs[-1:]
    err_l=err_l[-1:]
    
    w2=(brute_f_list[0][0]-brute_f_list[0][1])/len(brute_f_list[0])

    precision=0.1
    func=f"init: {w2}x^2 +f(x)"
    

    #__initial rss values required for model quality determination__

    temp=[ w2*(sq_ft[_]**2)+brute_f_list[0][_] for _ in range(len(sq_ft)) ]
    
    RSS=std_dev(price,temp)
    w2+=precision

    funcs[0]=(func )
    brute_f_list[0]=temp
    err_l[0]=RSS

    
    for _ in range(  1,stepsize+1  ):

        temp=[]
        
        for x in range(len(sq_ft)):
            
            out=w2*( sq_ft[x] **2) + brute_f_list[0][x]
            temp.append(out)

        ei=std_dev(price,temp)

        #if less error than function before, change RSS to new min
        #print(f"step{_} f:{mean_slope}x |prec:{precision} estimated e:{ei} Rss={RSS}")
        if ei<RSS:
            RSS=ei
            """functions can append here
            thus it will be more mem efficient
            but we can not see all steps on graph"""

            #if estimated RSS worse than function before, change direction
        else:
            #print(f"  ->step:{_} change direction,because {ei}>=RSS ")
            RSS=ei
            precision*=-0.5#reverts the slope direction
            """cahnge direction and degrade precision
            to estimate better model results """
        w2+=precision
        func=f"step:{_+1} {w2}x^2 +f(x)"
        funcs.append(func)
        brute_f_list.append(temp)
        err_l.append(ei)

def haloLearn(stepsize:int):
    global price
    global sq_ft
    
    global funcs
    global err_l
    global brute_f_list

    precision=0.1#func learning precision
    
    dy=mean(price)
    dx=mean(sq_ft)

    mean_slope=0.0122#dy/dx
    w0=0
    func=f"init: x^2 +{mean_slope:.4f}x+{w0}"
    

    #__initial rss values required for model quality determination__
    temp=[]
    c=0
    for x in sq_ft:
        c+=1
        out=f(x=x, a=mean_slope,b=w0)
        temp.append(out)

    RSS=std_dev(price,temp)
    mean_slope+=precision

    funcs.append(func )
    brute_f_list.append(temp)
    err_l.append( RSS )
    #____________________________________________________________
    print("initial func:",func,"rss=",RSS)

    
    for _ in range(  stepsize  ):

        temp=[]
        
        # find mean constant with y-ax=b on every change of w1 coeff
        w0=(dy- mean_slope*dx)
        
        for x in sq_ft:
            out=f(x=x, a=mean_slope,b=w0)
            temp.append(out)

        ei=std_dev(price,temp)

        #if less error than function before, change RSS to new min
        #print(f"step{_} f:{mean_slope}x |prec:{precision} estimated e:{ei} Rss={RSS}")
        if ei<RSS:
            RSS=ei
            """functions can append here
            thus it will be more mem efficient
            but we can not see all steps on graph"""

        #if estimated RSS worse than function before, change direction
        else:
            #print(f"  ->step:{_} change direction,because {ei}>=RSS ")
            RSS=ei
            precision*=-0.5#reverts the slope direction
            """cahnge direction and degrade precision
            to estimate better model results """
            
        mean_slope+=precision
        func=f"step:{_+1} {mean_slope:.4f}x^2 +{mean_slope:.4f}x+{w0}"
        funcs.append(func)
        brute_f_list.append(temp)
        err_l.append(ei)


sq_ft=[2, 4, 6, 12, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]
price=[2, 2,4 ,3, 4, 5 , 9,7,8, 10, 15 ,14 ,16, 18, 29, 32,35,39 ,36, 55, 59, 63, 64, 65, 72, 109, 111, 124 ,127, 125, 126]# Y=f(x)

if __name__=="__main__":
    fig1.canvas.manager.set_window_title('simple linear regression:haloLearn approach')

    fig1.clf()
    plt.scatter(sq_ft,price)
    #plt.xlim(0,34)
    #plt.ylim(0,28)
    print("Example:multipleReg. haloLearn\nrun with step size=16")
    haloLearn(16)
    multi_f(16)
    
    for b,f,e in zip(brute_f_list,funcs,err_l):
        
        plt.plot(sq_ft, b, "--",label=f"y={ f } err:{e:.3f}")
        fig1.legend(loc='upper left')
        plt.pause(0.5)
    #fig1.legend(loc='upper left')
    plt.show()
