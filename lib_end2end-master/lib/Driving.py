
import scipy.misc
import random

class Driving_Data(object):

    ####### DEFIND INSTAN #########
    NAME_FILE=None
    RADIANS_STEER=list()
    NAME_PITRUE=list()
    X,Y=None,None
    BATCH_POINTER=0

    def __init__(self,name_file):
        self.NAME_FILE=name_file
        self._Read_text()

    
    def Train(self,x1,x2):

        train_x=self.X[:int(x1)]
        train_y=self.Y[:int(x1)]
        
        val_x=self.X[int(x2)]
        val_y=self.Y[int(x2)]

        return {
            "train":[train_x,train_y],
            "val":[val_x,val_y]
        }


 


    @property
    def __Shuffle(self):

        _C=list(zip(self.NAME_PITRUE,self.RADIANS_STEER))
        random.shuffle(_C)
        self.X,self.Y=zip(*_C)
        train_instane=len(self.X)*0.8
        val_instane=len(self.X)*0.2
        Data_Train=self.Train(train_instane,val_instane)
   
        return Data_Train

    @property
    def __Number_image():
        return len(NAME_PITRUE)

    @staticmethod
    def Steering_wheel_angle(n):
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        return float((n.split()[1]).split(",")[0])* scipy.pi /180.0
        
         
    def LoadTrain_Batch(self ,batch_size):
        x_out,y_out=list(),list()
    
        for i in range(0,batch_size):
            
            x_out.append(scipy.misc.imresize(scipy.misc.imread(self.X[(self.BATCH_POINTER+i) ]% len(self.__Shuffle["train"][0]))[-150:],[66,200]) / 255.0)
            y_out.append([self.__Shuffle["train"][1][(self.BATCH_POINTER+i) % len(self.__Shuffle["train"][0])]])
        self.BATCH_POINTER+=batch_size
        return x_out,y_out
    
    def _Read_text(self):
        with open(self.NAME_FILE) as file:
            for i in file:
                self.NAME_PITRUE.append("data/"+i.split()[0])
                self.RADIANS_STEER.append(self.Steering_wheel_angle(i))
        

if __name__ == "__main__":
    dir='data.txt'
    obj=Driving_Data(dir)
    print(obj.LoadTrain_Batch(60))