from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from utils import *#my function definitions
from simple_linear_regression_BruteForce import *
from haloLearn import HaloLearn
#
#
#
#polinomial regression with gradient descent algorithm
class GradientDescend(HaloLearn):
    def __init__(self,train_x, train_y,functionOrder=2):
        super().__init__(train_x, train_y)

        self.RSS_prev=0#previous cost
        self.functionOrder=functionOrder
        self.LearningRate=1e-8    #step precision
        self.w=[]
        self.d_w=[]

        self.stepSize=2000
        
    def run(self):
        """calculates (a2)x^2 + f(x)=ax +b"""
        
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

        k=1         # direction coeff

        self.w=[0 for x in range(self.functionOrder+1)]
        #derivative of ERROR over wi(all weights). Gives us Gradients
        self.d_w=[0 for x in range(self.functionOrder+1)]
        
        func="init: "
        for _ in range(self.functionOrder,0,-1):
            func+=f"{self.w[_]:.4f}x^{_} +"
        func+=f"{self.w[0]:.4f} +"
        
        len_sqft=len(self.data.x)
        S=(-2/len_sqft )
        
        #MAIN LOOP (will train model as stepsize)
        for _step in range(  1,self.stepSize+1  ):

            temp=[]
            #calculate function according to coeffs
            for x in range(len_sqft):
                idx=x
                x=self.data.x[x]
                
                #predicted function:  wn*x^n  +w(n-1)*x^(n-1)...+w0
                out=0
                for _ in range(self.functionOrder,0,-1):
                    out+=f(x=x, a=self.w[_] ,power=_)
                out+=f(x=0, b=self.w[0])
                
                temp.append( out )#keep predicted outputs
                
                ei=std_dev_non_sqrt( [ self.data.y[idx] ],[out])#SUM(y-yi)/n
                
                for _ in range(self.functionOrder,0,-1):
                    self.d_w[_]= (x**_)*ei*S
                self.d_w[0]=ei
                
                #RSS=ei
                #new coefficients has calculated by gradient descent magnitude
                
                for _ in range(self.functionOrder,0,-1):
                    self.w[_]-=(1e-8*self.d_w[_])
                self.w[0]-=(1e-8*self.d_w[0])

                self.d_w=[0 for _ in range(self.functionOrder+1)]#clear derivative list


            ei=std_dev(self.data.y,temp)#mean squared error(cost) of trained model

            if abs(ei-self.RSS_prev)<1e-4:
                '''REACHED LIMITS OF DELTA RSS:'''
                print('break at:',_step,'| DELTA RSS=',(ei-self.RSS_prev))
                break;

            #print("new coefficients:",w2,w1,w0,"RSS:",ei)
            #input("devam")
            func=""
            for _ in range(self.functionOrder,0,-1):
                func+=f"{self.w[_]:.4f}x^{_} +"
            func+=f"{self.w[0]:.4f}"
            
            funcs.append(func)
            brute_f_list.append(temp)
            err_l.append(ei)
            
            self.RSS_prev=ei#keep ei as last estimated cost
            print("Step:",_step,"RSS Error:",self.RSS_prev)
        self.final()
        
    def final(self):
        pass


#sample dataset as second order polynomial
sq_ft=[x for x in range(2,202,2)]
price=[(.05142857*x)**2 +3.141592-(0.0859*x) for x in range(2,202,2)]

def main():
    fig1.canvas.manager.set_window_title('polinomial regression:gradient descend approach')

    plt.clf()
    plt.scatter(sq_ft,price,c="orange")
    plt.xlim(0,max(sq_ft)+5)
    plt.ylim(0,max(price)+5)
    print("Example:poliReg. with gradientDescent\nrun with step size=16")
    print("-"*40,"\n!!! IMPORTANT !!!\n","-"*40)
    print("if error rate changing badly, This means the precision(learning rate)\nchoosed too big for this set")
    print("-"*40,"\n","-"*40)
    GD=GradientDescend(sq_ft,price)
    GD.run()

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
if __name__=="__main__":
    main()
