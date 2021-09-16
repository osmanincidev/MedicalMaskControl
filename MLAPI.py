import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import csv
import re
from sklearn.metrics import accuracy_score
import math# to have squareroot of the dataset
class MachineL():
    def __init__(self):
        self.accuracy=-1
        self.split=0
        try:
            Dataset = pd.read_csv('data/Dataset.csv')
        except:
            Dataset=""#to have 0 length of training_data 
        if len(Dataset)>9:#becaseu 2 of ten is test and 8 out ten is training. so minimum I need 10 samples
            array=Dataset.values
            test_data=pd.read_csv('data/Dataset.csv')
            lastpic_data = pd.read_csv('data/LastPhoto.csv')
            lastpicArr=lastpic_data.values
            LastphotoArr=[] 
            arrayTest=test_data.values
            self.split=int(Dataset.shape[0]*20/100)
            self.split=Dataset.shape[0]-self.split#last 20 percent of data 
            
            X_train=array[:self.split,0:9215]#we have 9215 columns(features) 
            y_train=array[:self.split,9216] #label of train 9216th column is 1 or 0 respectively mask or not mask
            X_test=arrayTest[self.split+1:,0:9215] #the test data must be 20 percent(features)
            y_test=arrayTest[self.split+1:,9216]   #label of test data.
            neighbors=int(math.sqrt(len(Dataset)))
            if neighbors%2==0:
                neighbors=neighbors-1
            knn = KNeighborsClassifier(n_neighbors=neighbors)#similar to two samples neighbor.
            knn.fit(X_train,y_train)
            try:
                self.prediction = knn.predict(lastpicArr[:,0:9215])
            except NameError:
                print(NameError)
                
            predictions = knn.predict(X_test)#the test part is predicted
            self.accuracy=accuracy_score(y_test,predictions)*100 #accuracy value 
            self.LastPhotoPix()
        else:
            self.LastPhotoPix()
    def LastPhotoPix(self):
        LastPhottPixels=[]
        with open('data/LastPhoto.csv','r') as OneRow:
            reader=csv.reader(OneRow,delimiter=',')
            next(reader)#one line of csv file has error that is why we saved 2 times. now we need to skip one
            for line in reader:
                if len(line)!=0:
                    
                    for px in line:
                        LastPhottPixels.append(px)
                    try:
                        LastPhottPixels.append(self.prediction)
                    except:
                        import random
                        LastPhottPixels.append(random.randint(0,1))
        OneRow.close()
        with open('data/LastPhoto.csv','w') as OneRow:
            writer = csv.writer(OneRow,delimiter=',')#  writes all pixel of an image to each column of csv file
            writer.writerow(LastPhottPixels)
        OneRow.close()


