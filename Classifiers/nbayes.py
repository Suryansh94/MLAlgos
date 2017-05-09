import math
import sys
import csv
import numpy as np
from sklearn import model_selection

def separate_data(data):
	input1=[]
	input2=[]
	for vector in data:
		if(vector[-1].strip()=='- 50000.'):
			input1.append(vector)
		elif(vector[-1].strip()=='50000+.'):
			input2.append(vector)
	return input1,input2
def loadCsvFile(file):
	line=csv.reader(open(file, "rb"))
	return list(line)

def process_data(data):
	nominal={}
	mean_list=[]
	std_list=[]
	l=len(data[0])
	for j in xrange(l-1):
		
		if j in list([0,5,16,17,18,24,30,36]):
			cont=[]
			for i in xrange(len(data)):
				cont.append(float(data[i][j]))	 
			std_list.append(np.std(cont))
			mean_list.append(np.mean(cont))
			
		else:
			mp={}
			for i in xrange(len(data)):
				if data[i][j].strip(' ') =='?':
					continue
				if data[i][j]  in mp.keys():
					mp[data[i][j]]+=1
				else:
					mp[data[i][j]]=1
			for k, v in mp.items():
				prob=float(float(v)/float(len(data)))
				mp[k]=round(prob,5) 
			nominal[j]=mp
	return nominal,mean_list,std_list

if __name__ == "__main__":
	no=0
	kf = model_selection.KFold(n_splits=10,shuffle=True)
	data = loadCsvFile("census-income-final.csv")
	data=np.asarray(data)
	for x in xrange(30):
		for train, test in kf.split(data):
			traindata= list(data[train])
			input1,input2 = separate_data(traindata)
			testdata=list(data[test])
			Accuracy=0			
			nominal1,mean_list1,std_list1 = process_data(input1)
			nominal2,mean_list2,std_list2 = process_data(input2)
			Accr=[]
			for i in xrange(len(testdata)):
				pwc1=0
				pwc2=0
				
				for j in xrange(len(testdata[i])-1):
					if j in list([0,5,16,17,18,24,30,36]):
						continue
						
					else:
						if(testdata[i][j].strip(' ')!='?'):
							if testdata[i][j] in nominal1[j]:
								pwc1=pwc1+math.log(nominal1[j][testdata[i][j]])

							if testdata[i][j] in nominal2[j]:
								pwc2=pwc2+math.log(nominal2[j][testdata[i][j]])
				
				pwc1 += math.log(float(len(input1))/float(len(input1)+len(input2)))
				pwc2 += math.log(float(len(input2))/float(len(input1)+len(input2)))
				if (pwc2>pwc1 and testdata[i][-1].strip()=='50000+.')or (pwc1>pwc2 and testdata[i][-1].strip()=='- 50000.'):
					Accuracy+=1
			Accr.append((float(Accuracy)/float(len(testdata))) * 100.0)
		print "epoch number: ",
		print x,
		print "mean accuracy: ",	
		print np.mean(Accr)


