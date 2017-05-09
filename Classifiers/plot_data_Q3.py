import matplotlib.pyplot as plt
import numpy as np
class1=[[3,3],[3,0],[2,1],[0,2]]
class2=[[-1,1],[0,0],[-1,-1],[1,0]]
class3=[[3,3],[3,0],[2,1],[0,1.5]]
class4=[[-1,1],[0,0],[-1,-1],[1,0]]
c1x = [item[0] for item in class1]
c1y = [item[1] for item in class1]
c2x = [item[0] for item in class2]
c2y = [item[1] for item in class2]
c3x = [item[0] for item in class3]
c3y = [item[1] for item in class3]
c4x = [item[0] for item in class4]
c4y = [item[1] for item in class4]
weight1=np.array([ 0.75020912 ,0.66624649 ,-1.15399125])
weight2=np.array([ 0.89205852 ,0.41252309 ,-1.17858476])
slope1=-1/0.891304347826
slope2=-1/0.79958463136
mean1=[ 0.875 ,0.75 ]
mean2=[ 0.875 , 0.6875]

# // y = m*x +c
# // c= y-m*x , c1 = 0.75+0.891304347826*(0.875)
# c2 = 0.6875+(0.875/0.79958463136)
px=np.linspace(-6,10,5,dtype=int).tolist()
py1=[]
py2=[]
py1_for_mse=[]
py2_for_mse=[]
for i in px:
    #w1x+w2y+w0
    value1=(slope1*i)+(0.75+(0.875/0.891304347826))
    value2=(slope2*i)+(0.6875+(0.875/0.79958463136))
    value1_for_mse=(-weight1[2]-weight1[0]*i)/weight1[1]
    value2_for_mse=(-weight2[2]-weight2[0]*i)/weight2[1]
    py1.append(value1)
    py2.append(value2)
    py1_for_mse.append(value1_for_mse)
    py2_for_mse.append(value2_for_mse)
n=raw_input("use data set 1 or 2\n")
if n=='1':
    # data set 1
    axis=plt.gca()
    axis.set_xlim(-3,5)
    axis.set_ylim(-3,5)
    plt.plot(px,py1,"--")  # data set 1 classifier
    plt.plot(px,py1_for_mse,'-')
    plt.scatter(c1x,c1y,c="red")
    plt.scatter(c2x,c2y,c="blue")
    plt.show()
else:
     plt.plot(px,py1,"--") # data set 2 classifier
     plt.plot(px,py2_for_mse,'-')
     plt.scatter(c3x,c3y,c="green")
     plt.scatter(c4x,c4y,c="blue")
     plt.show()
