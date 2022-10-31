from math import sqrt
"""halilibo mut
my simple linear regression functions"""
#
#
#
#simple linear solution function a*x+b
def f(x, a=1, b=0,power=1):
    """generates linear function by given a,b,x values and returns result of x"""
    y=lambda x:a*(x**power)+b
    return y(x)

def mean(data:list):
    """calculates the mean of given dataset"""

    summary=0
    for a in data:
        summary+=a
    return summary/len(data)

def std_dev_mean(xi:list,mean):
    """calculates error rate sums of given list xi, by given mean of dataset"""
    l=len(xi)
    sum_error=0
    for a in range(l):
        
        val=((xi[a]-mean)**2 )/l
        sum_error+=val
    return sqrt(sum_error)

def std_dev(x:list,xi:list):
    """calculates error rate sums of estimated list xi, by given list(dataset) x"""
    l=len(x)
    sum_error=0
    for a in range(l):
        
        val=((x[a]-xi[a])**2 )/l
        sum_error+=val
    return sqrt(sum_error)

def cis(err_l:list,num_elements=6):
    """finds minimum indexes of error rates between functions"""
    indexes=[]
    
    for a in range(len(err_l)):
        
        if len(indexes)<num_elements:
            indexes.append( a )
            
        else:
            for element in indexes:

                if err_l[a] <err_l[element] and a not in indexes:
                    indexes.remove(element)
                    indexes.append(a)
    return indexes
#
#
#
#
#
def mxy(l1,l2):
    """ finds SUM(x*y)"""

    s=0
    for a,b in zip(l1,l2):
        s+=a*b
    return s

def m2(l):
    """finds SUM(x*x)"""
    s=0
    for a in l:
        s+=a*a
    return s
        
def coeff_x(mean_x,mean_y,mean_xy,mean_x2,n):
    """determines coefficient of x """

    a=(n*mean_xy -(mean_x*mean_y) )/ (n*mean_x2 -(mean_x*mean_x))
    return a

def const_x(mean_x,mean_y,slope,n):
    b=(mean_y- slope*mean_x)/n
    return b
if __name__=="__main__":
    print("utils for simple calcs")
