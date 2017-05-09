import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def votedpercep(data,it):
	n=len(data[0])
	wt_vectors=[]
	count_vector=[]
	count=0
	w=[]
	w_final=[]
	for i in range(0,n):
		w.append(float(0))
		w_final.append(float(0))
	w_final=numpy.array(w_final)
	w=numpy.array(w)
	eta=0.25
	classes=numpy.array(data)
	data_len=len(classes)
	wt_vectors.append(w)
	count_vector.append(1)
	while(it>=0):
		for i in range(0,len(classes)):
			val=numpy.dot(w,classes[i])
			if val<=0:
				w=w+(eta*classes[i])
				count+=1
				count_vector.append(1)
				wt_vectors.append(w)
			else:
				count_vector[count]+=1
		it-=1
	for i in range(0,len(count_vector)):
		w_final=w_final+count_vector[i]*wt_vectors[i]
	return w_final

def perceptron(class_final,it):
	n=0.5
	count=0
	iteration=0
	w=[]
	for i in range(0,len(class_final[0])):
		w.append(float(0))
		# w_final.append(float(0))
	class_final=numpy.array(class_final)
	w=numpy.array(w)
	while(it>=0):

		for i in range(0,len(class_final)):
			val=numpy.dot(w,class_final[i])
			if val<=0:
				w = w + n*class_final[i]
				count=0
				iteration=iteration+1
			else:
				count=count+1

		it-=1
	# print iteration
	# print w
	return w

def check(test,w):
	n=len(test)
	n=float(n)
	count=0.0
	w=numpy.array(w)
	test=numpy.array(test)
	for i in range(0,len(test)):
		val=numpy.dot(w,test[i])
		if val<=0:
			# print "here"
			# print test[i]
			count=count+1
	acc=(1-float(count/n))*100
	return float(acc)

dataset1=[]
f=open("ionosphere.txt","r")
for line in f:
	d=[]
	label=0
	d=line.split(',')
	d[len(d)-1]=d[len(d)-1].split('\n')[0]
	if d[len(d)-1]=='b':
		d[len(d)-1]=1
	elif d[len(d)-1]=='g':
		d[len(d)-1]=1
		label=-1
	d.pop(0)
	for i in range(0,len(d)):
		d[i]=float(d[i])
		if(label==-1):
			d[i]=0.0-d[i]
	dataset1.append(d)

dataset2=[]
f=open("breastcancer.txt","r")
for line in f:
	d=[]
	d=line.split(',')
	label=0
	d[len(d)-1]=int(d[len(d)-1].split('\n')[0])
	if d[len(d)-1]==2:
		label=-1
	else:
		label=1
	for i in range(0,len(d)):
		d[i]=float(d[i])
		if(i==len(d)-1):
			d[i]=d[i]-3.0
		elif(label==-1):
			d[i]=0.0-d[i]
	dataset2.append(d)

for i in dataset2:
	i.pop(0)


epochs_array=[10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
accuracy=[]

data=numpy.array(dataset2)
kf = KFold(n_splits=10)
for i in epochs_array:
	a=[]
	for train,test in kf.split(data):
		data_train, data_test= data[train], data[test]
		#call accordingly voted or normal perceptron here
		w=votedpercep(data_train,i)
		acc=check(data_test,w)
		a.append(acc)
	acr=float(sum(a)/len(a))
	accuracy.append(acr)
	# print w

# print accuracy
#Plot epoch and accuracy
plt.plot(epochs_array,accuracy,label='-',color='blue')
plt.show()
