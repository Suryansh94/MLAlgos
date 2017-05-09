import os
import sys
from scipy.misc import imresize
import numpy as np

n_input=64
n_hidden=10
n_output=10
eta=0.01
np.random.seed(0)

def resize_data(img):
	M=imresize(img,(8,8),interp='bicubic')
	M=M.flatten()
	bias=np.array([1])
	for i in range(64):
		if(M[i]>0):
			M[i]=1
	return M
	#return np.append(M,bias)

def sigmoid(X):
	return 1.0/(1.0 + np.exp(-X))

def der_sigmoid(X):
	return np.exp(-X)/((1.0+np.exp(-X))**2)

def train(x, y, V, W, bv, bw):

    #forward
	x=np.array(x)
	A = np.dot(x, V) + bv
	Y = sigmoid(A)

	B = np.dot(Y, W) + bw
	Z = sigmoid(B)

	err=np.subtract(y,Z)
	total_error.append(np.mean(np.abs(err)))

	#backward
	delta_Z = err*der_sigmoid(Z)
	hidden_error = np.dot(W,delta_Z)
	delta_Y = hidden_error*der_sigmoid(Y)

	#updation
	W+=eta*np.dot(np.matrix(Y).T,np.matrix(delta_Z))	#hidden to output
	V+=eta*np.dot(np.matrix(x).T,np.matrix(delta_Y))  #input to hidden
	return Z

def classify(x,V,W,bv,bw):
	A=np.dot(x,V) + bv
	B=np.dot(sigmoid(A),W) + bw
	C=sigmoid(B)
	return C

inps=[]
labels=[]
count=0
V = np.random.normal(scale=0.1, size=(n_input, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_output))
bv = np.zeros(n_hidden)
bw = np.zeros(n_output)
total_error=[]
inp=[]

f=open("optdigits-orig.tra","r")
for line in f:
		temp=[]
		label=[]
		for i in range(10):
			label.append(0)
		l=-1
		for i in line.strip():
			temp.append(int(i))
		if(len(temp) != 2):
			# temp.pop(len(temp)-1)
		 	inp.append(temp)
		if(len(temp) < 32):
			l=(int(line))
			label[l]=int(1)
			inp.pop(len(inp)-1)				
			D=resize_data(np.array(inp))
			labels.append(label)
			inps.append(D)
			inp=[]


for epoch in range(0,200):
	for i in range(len(inps)):
		z=train(inps[i],labels[i],V,W,bv,bw)
	if epoch%10==0 :
		print epoch
		print np.mean(np.abs(np.subtract(label,z)))		#Mean of error at the end of each epoch
	

f1=open("mindigits.txt","r")
for line in f1:
		temp=[]
		label=[]
		for i in range(10):
			label.append(0)
		l=-1
		for i in line.strip():
			temp.append(int(i))
		if(len(temp) != 2):
		 	inp.append(temp)
		if(len(temp) < 32):
			l=(int(line))
			label[l]=int(1)
			inp.pop(len(inp)-1)				
			D=resize_data(np.array(inp))
			ans=classify(D,V,W,bv,bw)
			print "actual label"
			print l
			print "obtained label"
			print np.argmax(np.array(ans))
			inp=[]