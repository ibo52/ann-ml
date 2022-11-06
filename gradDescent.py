from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from utils import *#my function definitions
"""halilibo mut"""

from simple_linear_regression_example_ANN import *

#polinomial regression with gradient descent algorithm
def gradientDesc(stepsize=16):
    """calculates (a2)x^2 + f(x)=ax +b"""
    global price
    global sq_ft
    
    global funcs
    global err_l
    global brute_f_list

    #best result of linear calculation done before
    brute_f_list=[]
    
    funcs=[]
    err_l=[]
    
    # the choice of the learning rate(precision), γ, is
    #important and has a significant impact
    #on the effectiveness of the algorithm.
    #https://www.freecodecamp.org/news/gradient-descent-machine-learning-algorithm-example/
    precision=1e-8    #step precision
    RSS_prev=0    #previous cost
    k=1         # direction coeff

    d_w2=[]
    d_w1=[]
    d_w0=[]

    w2=0
    w1=0
    w0=0
    func=f"init: {w2:.4f}x^2 +{w1}x +{w0}"
    
    len_sqft=len(sq_ft)
    S=(-2/len_sqft )
    for _ in range(  1,stepsize+1  ):

        temp=[]

        #calculate function according to coeffs
        for x in range(len_sqft):
            idx=x
            x=sq_ft[x]
            
            #predicted val:  a*x^2  +b*x         +c
            out=f(x=x,a=w2,power=2) +f(x=x,a=w1) +f(x=0,b=w0)
            temp.append( out )#predicted outputs
            
            ei=std_dev_non_sqrt( [ price[idx] ],[out])#SUM(y-yi)/n
            
            d_w2.append( (x**2)*ei )#derivative according to w2
            d_w1.append( x*ei )#    w1
            d_w0.append( ei )   #w0
            
            #RSS=ei

            #gradients
            d_w2=S*sum(d_w2)
            d_w1=S*sum(d_w1)
            d_w0=S*sum(d_w0)

            #new coefficients
            w2=w2-precision*d_w2
            w1=w1-precision*d_w1
            w0=w0-precision*d_w0

            d_w2=[]
            d_w1=[]
            d_w0=[]
            
        ei=std_dev(price,temp)#mean squared error(cost)
        if abs(ei-RSS_prev)<1e-3:
            '''REACHED LIMITS OF DELTA RSS:'''
            print('break at:',_,'| DELTA RSS=',(ei-RSS_prev))
            break;

        #print("new coefficients:",w2,w1,w0,"RSS:",ei)
        #input("devam")
        func=f"step:{_+1} {w2:.4f}x^2 +{w1:.4f}x +{w0:.4f}"
        funcs.append(func)
        brute_f_list.append(temp)
        err_l.append(ei)
        
        RSS_prev=ei#keep ei as last estimated cost


#sample dataset
sq_ft=[2, 4, 6, 12, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]
price=[2, 2,4 ,3, 4, 5 , 9,7,8, 10, 15 ,14 ,16, 18, 29, 32,35,39 ,36, 55, 59, 63, 64, 65, 72, 109, 111, 124 ,127, 125, 126]# Y=f(x)

if __name__=="__main__":
    fig1.canvas.manager.set_window_title('polinomial regression:gradient descend approach')

    plt.clf()
    plt.scatter(sq_ft,price)
    plt.xlim(0,max(sq_ft)+5)
    plt.ylim(0,max(price)+5)
    print("Example:poliReg. with gradientDescent\nrun with step size=16")
    print("-"*40,"\n!!! IMPORTANT !!!\n","-"*40)
    print("if error rate changing badly, This means the precision(learning rate)\nchoosed too big for this set")
    print("-"*40,"\n","-"*40)
    gradientDesc(2000)

    a=input('0. show steps(functions that changes over coeffs)\n1. show last step(best function estimated)\nChoice: ')

    if a=='0':
        pstep=7
        print(f"Last {pstep} steps will be figured")
        for b,f,e in zip(brute_f_list[-pstep:],funcs[-pstep:],err_l[-pstep:]):
        
            plt.plot(sq_ft, b, "--",label=f"y={ f } err:{e:.3f}")
            fig1.legend(loc='upper left')
            plt.pause(0.5)
    else:
        plt.plot(sq_ft, brute_f_list[-1], "--",label=f"y={ funcs[-1] } err:{err_l[-1]:.3f}")
        plt.legend()

        #save image
        plt.savefig(fname='out imgs/gradientDescend polinomial_example.jpg', dpi=300, format='jpg')
        
    plt.show()
