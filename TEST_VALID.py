from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from utils import *#my function definitions
"""halilibo mut"""

from simple_linear_regression_example_ANN import fig1

#polinomial regression with gradient descent algorithm
def gradientDesc(sq_ft=[],price=[], stepsize=16,power=2,precision=1e-8):
    """generates polinomial model at order of given power"""
    #global price
    #global sq_ft
    
    global funcs
    global err_l
    global brute_f_list

    
    brute_f_list=[]#list of last best outputs of models estimated
    funcs=[]#list of last best functions of models estimated
    err_l=[]#list of last best errors of models estimated
    
    # the choice of the learning rate(precision), γ, is
    #important and has a significant impact
    #on the effectiveness of the algorithm.
    #https://www.freecodecamp.org/news/gradient-descent-machine-learning-algorithm-example/
    #precision=1e-8    #step precision. setted as function argument
    RSS_prev=0    #previous cost
    k=1         # direction coeff

    #w0, w1, w2, w3...wn
    w_list=[0 for x in range(power+1)]

    #derivative of ERROR over wi(all weights). Gives us Gradients
    d_w_list=[0 for x in range(power+1)]

    #init function as text
    func="init: "
    for _ in range(power,0,-1):
        func+=f"{w_list[_]:.4f}x^{_} +"
    func+=f"{w_list[0]:.4f} +"

    len_sqft=len(sq_ft)
    S=(-2/len_sqft )

    #MAIN LOOP (will train model as stepsize)
    for _step in range(  1,stepsize+1  ):

        temp=[]

        #train model by given dataset
        for x in range(len_sqft):
            idx=x
            x=sq_ft[x]
            
            #predicted function:  wn*x^n  +w(n-1)*x^(n-1)...+w0
            out=0
            for _ in range(power,0,-1):
                out+=f(x=x, a=w_list[_] ,power=_)
            out+=f(x=0, b=w_list[0])
            
            temp.append( out )#keep predicted outputs
            
            ei=std_dev_non_sqrt( [ price[idx] ],[out])#RSS = SUM(y-yi)/n

            #calculate derivative of Cost function by wi(all weights)
            #to determine Gradient descent
            for _ in range(power,0,-1):
                d_w_list[_]= (x**_)*ei*S
            d_w_list[0]=ei
            
            #RSS=ei

            #new coefficients has calculated by gradient descent magnitude
            for _ in range(power,0,-1):
                w_list[_]-=( precision*d_w_list[_] )
            w_list[0]-=( precision* d_w_list[0])

            d_w_list=[0 for _ in range(power+1)]#clear derivative list
            
        ei=std_dev(price,temp)#mean squared error(cost) of trained model
        
        if abs(ei-RSS_prev)<1e-4:
            '''REACHED LIMITS OF DELTA RSS: MEANS the current trained model
            have very close outputs to last trained models. Thus no need
            to re-train anymore'''
            print('break at:',_step,'| DELTA RSS=',(ei-RSS_prev))
            break;

        #new function of model as text
        func=f"step:{_step+1} "
        for _ in range(power,0,-1):
            func+=f"{w_list[_]:.4f}x^{_} +"
        func+=f"{w_list[0]:.4f}"
        
        funcs.append(func)
        brute_f_list.append(temp)
        err_l.append(ei)
        
        RSS_prev=ei#keep ei as last estimated cost

    return w_list,funcs[-1],err_l[-1],temp

def train(x,y,splitRate=.8):
    len_x=len(x)# len_x ==len_y
    if splitRate>1 or splitRate<0:
        print("ERROR: split rate must be between [0,1]")
        exit("ERROR: split rate must be between [0,1]")
    
    TEST=int(len_x*splitRate)#first 80 element index
    VALID=(len_x - TEST)#last remained indexes

    print(f"splitting into %{splitRate*100} test and %{(1-splitRate)*100:.3f} validation set")
    w,func,test_err,test_out =gradientDesc(x[:TEST],y[:TEST], power=2)

    temp=[]
    for var in x[-VALID:]:
           
        out=0
        for _ in range(len(w)-1,0,-1):
            out+=f(x=var, a=w[_],power=_)
        out+=f(x=0, b=w[0])
           
        temp.append(out)

    VALID_ERR=std_dev(y[-VALID:], temp)
    print('-'*40)
    print(f"model: {func}\nTEST_ERR:{test_err}")
    print("trained model VALIDATION_ERR:",VALID_ERR)
    print('-'*40)
           
    plt.clf()
    # model over test set
    plt.scatter(x[:TEST],y[:TEST],color="yellow")
    plt.plot(x[:TEST], test_out,"--",color="blue",label=f"model:{func}\ntest_set err:{test_err}")

    #model over validation set
    plt.scatter(x[-VALID:],y[-VALID:],color="orange")
    plt.plot(x[-VALID:], temp,"--",color="red",label=f"model predict on valid set err:{VALID_ERR}")

    plt.legend()
    plt.savefig(fname=f'out imgs/test-valid(split:{splitRate}).jpg', dpi=300, format='jpg')
    plt.show()

#from data import price,sq_ft
#sample dataset
sq_ft=[x for x in range(2,202,2)]
price=[(.05142857*x)**2 +3.141592-(0.0859*x) for x in range(2,202,2)]

if __name__=="__main__":
    fig1.canvas.manager.set_window_title('polinomial regression:gradient descend approach')

    plt.clf()
    plt.scatter(sq_ft,price,color="orange")
    
    plt.xlim(min(sq_ft)-5,max(sq_ft)+5)
    plt.ylim(min(price)-5,max(price)+5)
    
    print("Example:poliReg. with gradientDescent\nSplitting dataset to test and validation sets")
    print("-"*40,"\n!!! IMPORTANT !!!\n","-"*40)
    print("if error rate changing badly, This means the precision(learning rate)\nchoosed too big for this set")
    print("-"*40,"\n","-"*40)
    train(sq_ft, price, splitRate=.2)
    input("now we will split data 65/35 .Press ENTER: ")
    train(sq_ft, price, splitRate=.65)
    
    exit(0)
