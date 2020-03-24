
import scipy.misc
import random

class Driving_Data(object):

    ####### DEFIND INSTAN #########
    NAME_FILE=None
    RADIANS_STEER=list()
    NAME_PITRUE=list()
    X,Y=None,None
    BATCH_POINTER=0
    
    PATH="D:/07012018/Autopilot-TensorFlow/"

    def __init__(self,name_file,**kwage):
        self.NAME_FILE=name_file
        self._Read_text()
        self.batch_size=kwage["batch_size"]
    
    def Train(self,x1,x2):
        result={}
        train_x=self.X[:int(x1)]
        train_y=self.Y[:int(x1)]
        
        val_x=self.X[-int(x2):]
        val_y=self.Y[-int(x2):]
        result["train"] =[train_x,train_y]
        result["val"] =[val_x,val_y]
        
        return result
        
    @property
    def __Shuffle(self):
        
        _C=list(zip(self.NAME_PITRUE,self.RADIANS_STEER))
        random.shuffle(_C)
        self.X,self.Y=zip(*_C)
        self. train_instane=len(self.X)*0.8
        self.val_instane=len(self.X)*0.2
        Data_Train=self.Train(self.train_instane,self.val_instane)
       
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
    
    @property
    def Getter_load_data(self):
        """
        return train 2 diaminion  and like 

        """
        return self.Load_Batch(self.__Shuffle["val"]),self.Load_Batch(self.__Shuffle["val"])

    def Load_Batch(self,resutl):
        """
        - แบ่งข้อมูลการ train 
        - Run ตาม batch size
        """
        x_out,y_out=list(),list()
        for i in range(0,self.batch_size):
           
            x_out.append(scipy.misc.imresize(scipy.misc.imread(resutl[0][(self.BATCH_POINTER+i) % len(resutl[0])])[-150:],[66,200]) / 255.0)
            y_out.append([resutl[1][(self.BATCH_POINTER+i) % len(resutl[0])]])
        self.BATCH_POINTER+=self.batch_size
        return x_out,y_out

    
    def _Read_text(self):
        """
         Read text file  from data set 
        """

        with open(self.NAME_FILE) as file:
            for i in file:
                self.NAME_PITRUE.append(self.PATH+"data/"+i.split()[0])
                
                self.RADIANS_STEER.append(self.Steering_wheel_angle(i))
        
"""
if __name__ == "__main__":
    dir='data.txt'
    obj=Driving_Data(dir,batch_size=60)
    train,val=obj.Getter_load_data
    print(val)
"""