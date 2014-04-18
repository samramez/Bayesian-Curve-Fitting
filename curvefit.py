import matplotlib
from numpy  import *
import numpy
import pytest
import matplotlib.pyplot as plt
from scipy import interpolate
import pylab as pl


#file = raw_input("enter the data file name in txt format (enter data.txt) \n")

# x = raw_input("enter the amount for the x (You can enter 10 as an exmple) \n")
x = 11
# M = raw_input("Please type in the highest order M (how about entering 4! ) \n")
M = 15

Mp = M + 1

N=[1,2,3,4,5,6,7,8,9,10]
#This is a counter for the Sigma seri 

alpha = 0.005
beta = 11.1


#start---------------------------------------------------------
#function to store the data in the matrix and returns Data stored in the text format file
def store(filename):
    file = open(filename)
    X = []
    for line in file.readlines():
        X.append(line)
    X = map(str.strip, X) #I use this to eliminate '\n' from my list
    return(X)
#end---------------------------------------------------------


#we are storing the file in Datas file. 
# Datas = store(file)
# Dlength = len(Datas)
Datas =[1222.52,1222.51,1222.29,1222.9,1222.37,1222.272,1222.356,1222.53,1222.69,1222.90]


#the following function, generates the PhiX
#start---------------------------------------------------------
def getPhiX(x):
    PhiX = []  #getting PhiX
    i = 0 
    while i < Mp:
        PhiX.append(math.pow(x,i))
        i += 1
    PhiX = numpy.array(PhiX)
    return PhiX
    
#end---------------------------------------------------------





# This function returns S^(-1)
#start---------------------------------------------------------
def getInvS():    
    I = numpy.ones((Mp,Mp)) #creating all one matrix as matrix I
    tmp = numpy.zeros((Mp,Mp)) # tmp = Phi(Xn).Phi(X)^T    
    n = 0
#     while n < len(N):
#         PhiX = getPhiX(N[n])
#         PhiX.shape(Mp,1)
#         PhiXT = numpy.transpose(PhiX) 
#         tmp += numpy.dot(PhiX,PhiXT) 
#         n += 1
        
    for n in range (0,len(N)):
        PhiX =getPhiX(N[n])  #Phix = PhiX[j]
        PhiX.shape=(Mp,1)
        PhiXT = numpy.transpose(PhiX) #DONE
        tmp += numpy.dot(PhiX,PhiXT) #DONE    
        
    betaPhi = beta*numpy.matrix(tmp) #multiplying a matrix by a constant number beta
    alphaI = alpha*numpy.matrix(I) #multiplying a matrix by a constant number alpha
    InverseS = alphaI + betaPhi
    return InverseS 
    # returns S^(-1)
#end---------------------------------------------------------





#This function returns mean(x)
#start---------------------------------------------------------
def getm(x):
    tmp = numpy.zeros( Mp ,1)
    PhiX = getPhiX(x)
    PhiX.shape= ( Mp ,1)
    InverseS = getInvS()
    S =  numpy.linalg.inv(InverseS) # getting inverse from InverseS and getting S
    PhiXT = numpy.transpose(PhiX)
    tmp1 = numpy.dot(PhiXT,S)
    mult = beta*numpy.matrix(tmp1)
    n = 0
    while n < len(N):
        phiXn = PhiX(N[n])
        z = numpy.dot(phiXn,Datas[n])
        tmp = numpy.add(tmp,z)
        n += 1
    result = numpy.dot(mult,z)
    return result



#end---------------------------------------------------------




#finding S^2
#start---------------------------------------------------------
def gets2(x):
    PhiX = getPhiX(x)
    PhiX.shape = (Mp,1)
    InverseS = getInvS()
    PhiXT = numpy.transpose(PhiX)
    S =  numpy.linalg.inv(InverseS)    
    tmp = numpy.dot(PhiXT,S)
    s2 = numpy.dot(tmp,PhiX)
    s2 += (1/beta)
    return s2

#end---------------------------------------------------------





#start---------------------------------------------------------
def showPlot():
    pl.plot(Datas)
    pl.show()
#end---------------------------------------------------------




def main():
    
    
    
    plt.figure(1)
    
    #we set the range of the plot
    range = numpy.arange(1215,1230,0.1)
    S2 = gets2(x)
    MeanX = getm(x)
    tmp = MeanX[0]
    s = S2[0][0]
    print(s)
    print (tmp)
#      plt.plot(range,yy)
#     plt.show()
    
    print
    fir = 1/(2*math.pi*s)
    fir = fir**0.5
    
    sec = 1/(2*s)
    sec = sec*((range-tmp)**2)
    yy = fir*(math.e)**(-sec)
    
    
#     print
#     fir = 1/(2*math.pi*s)
#     fir = fir**0.5
#     plt.ylim(0,0.5)
    
    plt.plot(range,yy)
    plt.show()
    

    
if __name__ == '__main__':
    main()
    
