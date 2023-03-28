from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from utils import *#my function definitions
from simple_linear_regression_BruteForce import *
#
"""halilibo mut"""
#
#
class HaloLearn(ModelBackBone):
    def __init__(self, x_train, y_train):
        super().__init__(x_train,y_train)
        
        self.learningRate=1  #func learning precision
        self.dx=0
        self.dy=0

        self.stepSize=16 #how many iteration will be performed

        self.w=[0]#
        
    def run(self):
        global price
        global sq_ft
        
        global funcs
        global err_l
        global brute_f_list
        print("dat",len(self.data.x), len(self.data.y))
        self.dy=mean(self.data.y)
        self.dx=mean(self.data.x)

        mean_slope=self.dy/self.dx
        self.w[0]=0


        print("initial function:",self.function,"rss=",self.RSS)
        #__initial rss values required for model quality determination
        #____________________________________________________________

        for _ in range(  self.stepSize  ):
            
            temp=[]
            
            # find mean constant with y-ax=b on every change of w1 coeff
            self.w[0]=(self.dy- mean_slope*self.dx)
            
            for x in self.data.x:
                out=f(x=x, a=mean_slope,b=self.w[0])
                temp.append(out)

            self.function=f"{mean_slope:.4f}x+{self.w[0]}"
            ei=self.calc_errors(temp)

            if abs(ei-self.RSS)<1e-4:
                '''REACHED LIMITS OF DELTA RSS:'''
                print('break at:',_,'| DELTA RSS=',(ei-self.RSS))
                break;
            #if less error than function before, change RSS to new min
            #print(f"step{_} f:{mean_slope}x |prec:{precision} estimated e:{ei} Rss={RSS}")
            if ei<self.RSS:
                self.RSS=ei
                """functions can append here
                thus it will be more mem efficient
                but we can not see all steps on graph"""

            #if estimated RSS worse than function before, change direction
            else:
                #print(f"  ->step:{_} change direction,because {ei}>=RSS ")
                self.RSS=ei
                self.learningRate*=-0.5#reverts the slope direction
                """cahnge direction and degrade precision
                to estimate better model results """
                
            mean_slope+=self.learningRate
            funcs.append(self.function)
            
            brute_f_list.append(temp)
            err_l.append(ei)

if __name__=="__main__":

    HL=HaloLearn(x,y)
    HL.run()
    
    fig1.canvas.manager.set_window_title('simple linear regression:haloLearn approach')

    print("Example:haloLearn\nrun with step size=16")

    for b,f,e in zip(brute_f_list,funcs,err_l):
        plt.plot(HL.data.x, b, "--",label=f"y={ f } err:{e:.3f}")
        fig1.legend(loc='upper left')
        plt.pause(0.5)

    #fig1.legend(loc='upper left')
    plt.show()
