
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[52]:

class1=[[3,3,1],[3,0,1],[2,1,1],[0,2,1]]
class2=[[1,-1,-1],[0,0,-1],[1,1,-1],[-1,0,-1]]
class3=[[3,3,1],[3,0,1],[2,1,1],[0,1.5,1]]
class4=[[1,-1,-1],[0,0,-1],[1,1,-1],[-1,0,-1]]
data1=class1+class2
data2=class3+class4


# In[53]:

c1x,c1y,c2x,c2y,c3x,c3y,c4x,c4y=[],[],[],[],[],[],[],[]
for i in class1:
    c1x.append(i[0])
    c1y.append(i[1])
for i in class2:
    c2x.append(-i[0])
    c2y.append(-i[1])
for i in class3:
    c3x.append(i[0])
    c3y.append(i[1])
for i in class4:
    c4x.append(-i[0])
    c4y.append(-i[1])


# In[78]:
weight1=np.array([0,0,0]) # w1,w2,w0 for first data set
weight2=np.array([0,0,0]) # for second deata set
b=2
k=1
flag=0
eta=0.2
ETA=eta
theta=1e-100
VALUE=2
while VALUE>theta:
    flag=0
    k+=1
    if k>1000:
        break
    for point in data1:
        y=np.array(point)
        val=np.inner(weight1,y)
        weight1=weight1+eta*(y.T*(b-val))
        eta=ETA/k
        VALUE=np.linalg.norm(eta*(y.T*(b-val)))
    # if flag==0:
    #     break

print "iterations for dataset1 = ",k,weight1


# k=1
# while VALUE>theta:
#     flag=0
#     k+=1
#     if k>1000:
#         break
#     for point in data2:
#         y=np.array(point)
#         val=np.inner(weight2,y)
#         weight2=weight2+0.2*(y.T*(b-val))
#         eta=ETA/k
#         VALUE=np.linalg.norm(eta*(y.T*(b-val)))
# #             flag=1
# #     if flag==0:
# #         break
# print "iterations for dataset2 = ",k,weight2


# In[80]:

# generating random points to plot line
px=np.linspace(-2,4,5,dtype=int).tolist()
py1=[]
py2=[]
for i in px:
    #w1x+w2y+w0
    value1=(-weight1[2]-weight1[0]*i)/weight1[1]
    value2=(-weight2[2]-weight2[0]*i)/weight2[1]
    py1.append(value1)
    py2.append(value2)
axis=plt.gca()
axis.set_xlim(-3,5)
axis.set_ylim(-3,5)
# uncomment to see for data set 1

plt.scatter(c1x,c1y,c="red",marker='o')
plt.scatter(c2x,c2y,c="yellow",marker='o')
plt.plot(px,py1,'-')

#comment the below two lines to see for dataset2

# plt.scatter(c3x,c3y,c="red",marker='^')
# plt.scatter(c4x,c4y,c="yellow",marker='^')
# plt.plot(px,py2,'-')
plt.show()
