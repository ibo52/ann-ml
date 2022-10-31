from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from utils import *#my function definitions
"""halilibo mut"""

from simple_linear_regression_example_ANN import *

#polinomial regression
def multi_f(stepsize,power=2):
    """calculates (a2)x^2 + f(x)=ax +b"""
    global price
    global sq_ft
    
    global funcs
    global err_l
    global brute_f_list

    #best result of linear calculation done before
    brute_f_list=brute_f_list[-1:]
    
    funcs=funcs[-1:]
    err_l=err_l[-1:]
    
    w2=0

    precision=1#step precision
    k=1# direction coeff
    func=f"init: {w2:.4f}x^2 +f(x)"
    

    #__initial rss values required for model quality determination__

    #w2=0 as initial value, so func will be linear(brute_f_list[0] ) on init
    RSS=std_dev(price,brute_f_list[0])
    w2+=precision

    funcs[0]=(func )
    err_l[0]=RSS

    
    for _ in range(  1,stepsize+1  ):

        temp=[]
        
        for x in range(len(sq_ft)):
            
            #deprecated calculation
            #out=w2*( sq_ft[x] **2) + brute_f_list[0][x]
            out= f(x=sq_ft[x], a=w2, b=0, power=power)
            temp.append(out)

        ei=std_dev(price,temp)
        precision=np.log(abs(precision) + abs(RSS-ei))
        #if less error than function before, change RSS to new min
        #print(f"step{_} f:{mean_slope}x |prec:{precision} estimated e:{ei} Rss={RSS}")
        if abs(ei-RSS)<1e-4:
            '''REACHED LIMITS OF DELTA RSS:'''
            print('break at:',_,'| DELTA RSS=',(ei-RSS))
            break;
        
        if ei<RSS:
            RSS=ei
            """functions can append here
            thus it will be more mem efficient
            but we can not see all steps on graph"""

            #if estimated RSS worse than function before, change direction
        else:
            #print(f"  ->step:{_} change direction,because {ei}>=RSS ")
            RSS=ei
            
            k*=-0.5 #change direction and lower the stepping range
            precision=np.e #reset stepping range precision
            
        w2+=precision*k
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

    precision=1#func learning precision
    k=1#direction coefficient
    
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

        #increase the velocity logarithmically as we on true direction
        precision=np.log(abs(precision) + abs(RSS-ei))
        
        #if less error than function before, change RSS to new min
        #print(f"step{_} f:{mean_slope}x |prec:{precision} estimated e:{ei} Rss={RSS}")
        if abs(ei-RSS)<1e-4:
            '''REACHED LIMITS OF DELTA RSS:'''
            print('break at:',_,'| DELTA RSS=',(ei-RSS))
            break;
        
        if ei<RSS:
            
            RSS=ei
            """functions can append here
            thus it will be more mem efficient
            but we can not see all steps on graph"""

        #if estimated RSS worse than function before, change direction
        else:
            #print(f"  ->step:{_} change direction,because {ei}>=RSS ")
            RSS=ei
            '''
            if precision==0:
                precision=2.71828
            if precision<0:
                precision=np.log(abs(precision))#reverts the slope direction
            else:
                precision=np.log(precision)
            '''
            k*=-0.5 #change direction and lower the stepping range
            precision=np.e#reset stepping range precision to e
            
            """change direction and reset precision
            to approach limit of last best model result """
            
        mean_slope+=precision*k
        func=f"step:{_+1} {mean_slope:.4f}x+{w0:.4f}"
        funcs.append(func)
        brute_f_list.append(temp)
        err_l.append(ei)


sq_ft=[2, 4, 6, 12, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]
price=[2, 2,4 ,3, 4, 5 , 9,7,8, 10, 15 ,14 ,16, 18, 29, 32,35,39 ,36, 55, 59, 63, 64, 65, 72, 109, 111, 124 ,127, 125, 126]# Y=f(x)

if __name__=="__main__":
    fig1.canvas.manager.set_window_title('polinomial regression:haloLearn approach')

    plt.clf()
    plt.scatter(sq_ft,price)
    plt.xlim(0,60)
    plt.ylim(0,120)
    print("Example:poliReg. haloLearn\nrun with step size=16")
    haloLearn(165)
    multi_f(1000)
    """
    for b,f,e in zip(brute_f_list,funcs,err_l):
        
        plt.plot(sq_ft, b, "--",label=f"y={ f } err:{e:.3f}")
        fig1.legend(loc='upper left')
        plt.pause(0.5)
    """    
    plt.plot(sq_ft, brute_f_list[-1], "--",label=f"y={ funcs[-1] } err:{err_l[-1]:.3f}")
    plt.legend()
    print(min(err_l))
    #fig1.legend(loc='upper left')
    plt.show()
