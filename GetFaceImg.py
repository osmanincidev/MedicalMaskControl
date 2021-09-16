import cv2
from tkinter import *
import csv
from PIL import Image
from PIL import ImageTk
import tkinter.font as font
   
class Camera():
    def __init__(self,theface,MLAPI):
        
        self.MLAPI=MLAPI
        self.root=Tk()
        self.root.configure(bg="Gray")
        self.theface=theface
        self.restart=0
        self.State=None
        self.faces=None
        self.grayFrame=None
        self.OneorZero=None
        self.yesNoflag=None #the user click yes or no to save pixels to do this only once
        self.plysnd=1
        self.acc=-1#when I dont have any dataset I wanna control some operations.
        self.NumOfMask=0
        self.NumOfNoMask=0
        self.TestMask=0
        self.TestNoMask=0
        self.root.geometry("850x400")
        self.thevideo=cv2.VideoCapture(0)#to use default camera.     
        self.trainbtn=Button(self.root,text="Train Me",bg="Green",command=self.trainFunc) 
        self.trainbtn.pack(side=LEFT,padx=20,ipadx=20,ipady=50)
        self.hirebtn=Button(self.root,text="Hire Me",bg="orange",command=self.hireFunc) 
        self.hirebtn.pack(side=LEFT,padx=20,ipadx=20,ipady=50)
        self.desc=Text(self.root,highlightbackground='gray',height=15,width=50,
                          font=font.Font(size=12),bg="#00bfff",fg="black")
        self.desc.config(wrap=WORD)
        self.desc.pack(side=LEFT,padx=20)
        intro=" Osman İNCİ \n 160101010 \n dvlpsmn@gmail.com \nMachine Learning Based Real time Medical Mask Control \n to expand dataset click train me or To use Hire me \n if the User decided to use to Train, each time the program capture a picture it will guess and write top of the image 'No Mask' or 'With Mask' and after the user click yes or no it will save pixels to Dataset \n The Accuracy will be writen down below of the image \n Also The User Must look Straight to the Camera.the face size will be 96X96 pixels. so depends on the size of image it is important to be closer or further"
        self.desc.insert(END,intro)        
        self.root.mainloop()
    def hireFunc(self):
        self.State=1#hire
        self.HireOrTrain()
    def trainFunc(self):
        self.State=0#train
        self.HireOrTrain()
        
    def HireOrTrain(self):
        self.trainbtn.destroy()
        self.hirebtn.destroy()
        self.desc.destroy()
        self.videoFrame=Frame(self.root)
        self.videoFrame.pack(side=LEFT)
        self.label=Label(self.videoFrame)#to show the video in the label
       
        self.BtnFrame=Frame(self.root,bg="green")
        self.BtnFrame.pack(side=LEFT)
        self.imgFrame=Frame(self.root,bg="red")
        self.imgFrame.pack(side=LEFT)
        self.Message=Text(self.imgFrame,highlightbackground='gray',height=1.4,width=10,
                 font=font.Font(size=11),bg="#00bfff",fg="black")
        self.Message.config(wrap=WORD)

        self.Samples=Text(self.imgFrame,highlightbackground='gray',height=6,width=14,
                          font=font.Font(size=8),bg="#00bfff",fg="black")
        self.Samples.config(wrap=WORD)                         
        
        self.AccyracyText=Text(self.imgFrame,highlightbackground='gray',height=1.4,width=10,
                 font=font.Font(size=11),bg="#00bfff",fg="black")       
        self.canvaspic=Canvas(self.imgFrame,width=96,height=96,bg="gray")
        self.YesBtn=Button(self.BtnFrame,text="Yes",width=5,command=self.Yes) 
        self.NoBtn=Button(self.BtnFrame,text="No",width=4,command=self.No) 
        self.SaveBtn=Button(self.BtnFrame, text="Save The Face",width=10,command=self.savTheFace,bg="green")
        self.RestartBtn=Button(self.BtnFrame,text="Restart",width=10,command=self.Restart,bg="green")
        #randomize button randomizes all the samples in Dataset.csv file
        self.RandomizeBtn=Button(self.BtnFrame,text="Randomize",width=10,command=self.RandomizeDataset,bg="green")
        self.soundbtn=Button(self.BtnFrame,text="Sound On",width=10,command=self.playsoundstate,bg="green")
        self.RestartBtn.pack()
        self.RandomizeBtn.pack()
        self.soundbtn.pack()
        if self.State==0:
            self.SaveBtn.pack(side=TOP)
        self.ShowTk()
    def ShowTk(self):
        success,img=self.thevideo.read()#img is each frame of the view
        self.grayFrame=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.faces=self.theface.detectMultiScale(self.grayFrame,scaleFactor=2,minNeighbors=3)
        #scalefactor each time how much size if reduced.%20 percent of the image
        #minNeighbors=3 number of neighbor to be kept by each rectangle in case we have wrong positive
        img=Image.fromarray(img)#img is np to use it in pillow we need it to show on label.
        img=ImageTk.PhotoImage(img)
        self.label.configure(image=img)
        self.label.image=img
        self.label.pack(side=LEFT)
        self.label.after(1,self.ShowTk)#delay for a ms and call ShowTk to show video in label
        if self.State==1:#in case the program is in hire mode.
            self.savTheFace()
    def savTheFace(self):#to save  frame of the face
        for x,y,w,h in self.faces:
           
            if h==96 & w==96:#to have 96X96 size image of the face
                roi_gray=self.grayFrame[y:y+h,x:x+w]#detected face
                nameOfImg="data/theFrame.png"
                cv2.imwrite(nameOfImg,roi_gray)
                self.SaveThePix()#saves the pixels to a csv file
           
                
    def SaveThePix(self):
        CapturedImg=Image.open("data/theFrame.png",'r')#to read pixels of the captured image(face)  
        ImgPixels=[]
        for col in range(CapturedImg.size[0]):
            for row in range(CapturedImg.size[1]):
                currentPixel=CapturedImg.getpixel((col,row))
                ImgPixels.append(currentPixel)
        with open('data/LastPhoto.csv','w+') as OneRow:
            writer = csv.writer(OneRow,delimiter=',')#  writes all pixel of an image to each column of csv file
            writer.writerow(ImgPixels)
        OneRow.close()
        #only one row have error in csv file while to make predict in knn that is why I append the same value again
        with open('data/LastPhoto.csv','a',newline="") as OneRow:
            writer = csv.writer(OneRow,delimiter=',')#  writes all pixel of an image to each column of csv file
            writer.writerow(ImgPixels)
        OneRow.close()
        self.Message.pack(side=TOP)#no mask or with mask text 
        self.AccyracyText.pack()
        self.Samples.pack()#information about dataset
        self.canvaspic.pack(side=LEFT)#the captured face image
        self.img=Image.open("data/theFrame.png")
        self.image=ImageTk.PhotoImage(self.img)
        self.canvaspic.create_image(0,0,anchor=NW,image=self.image)
        self.yesNoflag=0#each time new image is captured the use can click one of buttons(yes or no to save pixels)
        self.machineLearning()

    def machineLearning(self):
        MLObj=self.MLAPI.MachineL()                
        self.AccyracyText.delete('1.0', END)
        self.AccyracyText.insert(END,"acc: %"+str(MLObj.accuracy))
        self.infoSamples(MLObj.split)
        self.decide(MLObj.accuracy)
    def infoSamples(self,split):
         #to calculate number of positive and negative samples
        self.NumOfMask=0
        self.NumOfNoMask=0
        self.TestNoMask=0
        self.TestMask=0
        count=0
        with open('data/Dataset.csv','r') as OneRow:
            reader = csv.reader(OneRow,delimiter=',')
            for line in reader:
                if len(line)!=0: 
                    if str(line[9216])=="1":#the last element of csv file (the Label)
                        self.NumOfMask=self.NumOfMask+1 #to calcuate positive(with mask) samples
                    else:
                        self.NumOfNoMask=self.NumOfNoMask+1#to calcuate negative(no mask) samples
                        
                    #the Dataset is splitted by 20%. to count if we arrived part of test 
                    #to calculate with mask and no mask in test part.
                    if count>split:
                        if str(line[9216])=="1":
                            self.TestMask=self.TestMask+1 #in the test side positive samples
                        elif str(line[9216])=="0":
                            self.TestNoMask=self.TestNoMask+1#in the test side negative samples
                    count=count+1
    def decide(self,acc):
        self.acc=acc
        if self.State==0:#those button appear in train part
            self.YesBtn.configure(bg="green")
            self.NoBtn.configure(bg="green")
            self.YesBtn.pack(side=LEFT)
            self.NoBtn.pack(side=LEFT)
        if acc!=-1:#when the dataset  is not empty    
            self.Samples.delete('1.0',END)
            numofmask="Mask:"+str(self.NumOfMask)#total number of positive samples
            self.Samples.insert(END,numofmask)
            numofNomask="\nNo Mask:"+str(self.NumOfNoMask)#total number of negative samples
            self.Samples.insert(END,numofNomask)
            numoftestmask="\nTest Mask:"+str(self.TestMask)#number of positive samples in test part
            self.Samples.insert(END,numoftestmask)
            numofTestNoMask="\nTest No Mask:"+str(self.TestNoMask)#number of negative samples in test part
            self.Samples.insert(END,numofTestNoMask)
            numOfTrainMask="\nTrn Mask:"+str(self.NumOfMask-self.TestMask)#number of positive samples in train part
            self.Samples.insert(END,numOfTrainMask)
            numofTrainNoMask="\nTrn No Mask:"+str(self.NumOfNoMask-self.TestNoMask)#number of negative samples in train part
            self.Samples.insert(END,numofTrainNoMask)
        else:
            self.Samples.delete('1.0',END)
            allsamples="All Samples:"+str(self.NumOfMask+self.NumOfNoMask)#total number of samples
            self.Samples.insert(END,allsamples)
            description="\nThe Prediction is done Randomly.\n this is total number of samples"#
            self.Samples.insert(END,description)
            

        #below check if the 9217th column to decide if mask or no mask. this column is filled by MLAPI file
        with open('data/LastPhoto.csv','r') as OneRow:
            reader = csv.reader(OneRow,delimiter=',')
            for line in reader:
                if len(line)!=0:
                    if str(line[9216])=="[1]":
                        self.OneorZero=1#it is the value that I m going to 
                        self.Message.delete('1.0', END)
                        self.Message.insert(END,"With Mask")
                        if self.plysnd==1:
                            self.playsound()
                    else:
                        self.OneorZero=0
                        self.Message.delete('1.0', END)
                        self.Message.insert(END,"No Mask")
                        if self.plysnd==1:
                            self.playsound()
        OneRow.close()
    def playsound(self):
        try:
            import playsound
            if self.OneorZero==0:
                playsound.playsound('data/WearMask.mp3',False)
            elif self.OneorZero==1:
                playsound.playsound('data/Thanks2Wear.mp3',False)
        except:
            pass
        
    def playsoundstate(self):
        self.plysnd=(self.plysnd+1)%2
        if self.plysnd==0:
            self.soundbtn.configure(text="Sound Off")
        else:
            self.soundbtn.configure(text="Sound On")
            
    def Yes(self):
        
        if self.yesNoflag==0:#the button yes button will be clicked only once on same photo
            self.YesBtn.configure(bg="red")
            lastphotopix=[]
            with open('data/LastPhoto.csv','r') as OneRow:
                reader = csv.reader(OneRow,delimiter=',')
                for line in reader:
                    if len(line)!=0:
                        for px in line:
                            lastphotopix.append(px)
                        lastphotopix[9216]=self.OneorZero
                        with open('data/Dataset.csv','a',newline="") as Dataset:
                            writer=csv.writer(Dataset,delimiter=',')
                            writer.writerow(lastphotopix)
                        Dataset.close()
                        self.yesNoflag=1
                        self.UpdateSample(self.OneorZero)#to increarse or decrase number of samples
    def No(self):
        
        if self.yesNoflag==0:#the button No button will be clicked only once on same photo
            self.NoBtn.configure(bg="red")
            lastphotopix=[]
            with open('data/LastPhoto.csv','r') as OneRow:
                reader = csv.reader(OneRow,delimiter=',')
                for line in reader:
                    if len(line)!=0:
                        for px in line:
                            lastphotopix.append(px)
                        lastphotopix[9216]=(self.OneorZero+1)%2
                        with open('data/Dataset.csv','a',newline="") as Dataset:
                            writer=csv.writer(Dataset,delimiter=',')
                            writer.writerow(lastphotopix)
                        Dataset.close()
                        self.yesNoflag=1
                        self.UpdateSample((self.OneorZero+1)%2)
    def Restart(self):
        self.root.destroy()
        self.restart=1
        
    def RandomizeDataset(self):
        import random
        DatasetArr=[]
        with open('data/Dataset.csv','r') as OneRow:
            reader = csv.reader(OneRow,delimiter=',')
            for line in reader:
                DatasetArr.append(line)
        OneRow.close()
        random.shuffle(DatasetArr)
        with open('data/Dataset.csv','w',newline="") as OneRow:
            writer = csv.writer(OneRow,delimiter=',')
            for line in DatasetArr:
                if len(line)!=0:
                    writer.writerow(line)
    def UpdateSample(self,value):
        if value==0:    
            self.NumOfNoMask=self.NumOfNoMask+1
        else:
            self.NumOfMask=self.NumOfMask+1
            
            
        
        
        
            
                
            
        
        










        

