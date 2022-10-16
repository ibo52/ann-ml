from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from utils import *#my function definitions
"""halilibo mut"""

from simple_linear_regression_example_ANN import *

def haloLearn(stepsize:int):
    global price
    global sq_ft
    
    global funcs
    global err_l
    global brute_f_list

    precision=1#func learning precision
    
    dy=mean(price)
    dx=mean(sq_ft)

    mean_slope=dy/dx
    w0=0
    func=f"init: {mean_slope:.4f}x+{w0}"
    

    #__initial rss values required for model quality determination__
    temp=[]
    for x in sq_ft:
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
        func=f"step:{_+1} {mean_slope:.4f}x+{w0}"
        funcs.append(func)
        brute_f_list.append(temp)
        err_l.append(ei)

if __name__=="__main__":
    fig1.canvas.manager.set_window_title('simple linear regression:haloLearn approach')

    print("Example:haloLearn\nrun with step size=16")
    haloLearn(16)
    
    for b,f,e in zip(brute_f_list,funcs,err_l):
        plt.plot(sq_ft, b, "--",label=f"y={ f } err:{e:.3f}")
        fig1.legend(loc='upper left')
        plt.pause(0.5)

    #fig1.legend(loc='upper left')
    plt.show()
