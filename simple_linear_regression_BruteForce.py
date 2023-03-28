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

brute_f_list=[]#estimated function results
funcs=[]#to print functions as strings
err_l=[]#error rate of function values


class ModelBackBone:
    def __init__(self,x_train,y_train):
        self.range=3.2

        self.data=Data(x_train, y_train)

        self.stepSize=0 #how many iteration will be performed
        self.w=[0]#coefficients of features
        self.learningRate=1  #func learning precision
        self.dx=0
        self.dy=0
        self.function=f"init: w0={self.w[0]}"

        #have to be infinite at initialize,we set all 32 bits to 1 (2^32 -1) 
        self.RSS=0xffffffff #mse function; residual sum of squares

        self.verbose=1# to print error on screen
        
        
    def run(self):
        """produces the linear solution models by given ranges and precision"""
        pass

    def calc_errors(self,train_out):
        global brute_f_list
        global err_l
        
        ei=std_dev(self.data.y,train_out)
        err_l.append(ei)

        if(self.verbose):
            print("Estimated error:",ei,"function:",self.function)
        return ei
    
class Data:
    def __init__(self,x_train,y_train):
        self.x=x_train
        self.y=y_train
        
    def _return(self):
        return (self.x,self.y)

#
#
#
class BruteForce(ModelBackBone):
    def __init__(self, x_train, y_train):
        super().__init__(x_train, y_train)
        print(self.learningRate)

    def run(self):
        """produces the linear solution models by given ranges and precision"""
        global brute_f_list
        global funcs
        
        self.ran=3.2#range of brute force
        self.learningRate=0.1#choose bigger since its brute forcing
        round_coeff=len( str(self.learningRate).split(".")[1] )
        
        for a in np.arange(0, self.range, self.learningRate):
            temp=[]
        
            for b in np.arange(0.1, self.range, self.learningRate):
                for x in self.data.x:
                    out=f(x=x, a=a, b=b)
                    temp.append(out)

                func=f"{a:.{round_coeff}f}x+{b:.{round_coeff}f}"
                funcs.append(func )
                self.function=func
                brute_f_list.append(temp)

                self.RSS=self.calc_errors(temp)
                temp=[]


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
plt.plot(x,fx,"o-",label=f"{a:.4f}x+{b:.4f} err:{std_dev(price,fx):.4f} (economics formula)")

#---approved solution by field of economics---

if __name__=="__main__":
    bruteForce=BruteForce(x,y)
    bruteForce.run()
    indexes=cis(err_l,5)#find the #(5) of indexes that have least error rates










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
