import math
import scipy.misc
import random

class Process(object):

    _X=list()
    _Y=list()

    def __init__(self,name):
        self.__name=name
        self.read()

    def read(self):
        with open(self.__name,'r') as line:
            for i in line:
                self._X.append("data/" + i.split()[0])
                l=i.split()[1]
                l=l.split(",")
                self._Y.append(float(l[0]) * math.pi  / 180)
    @property
    def GetValue(self):
        return  self._X ,self._Y




class Train_Test(Process):

    _batch_pointer=0

    def __init__(self,**kwarge):
        
        # To change the train and test ratio do it here
        # Train -> 80% and Test -> 20% good for u 
        self._traing=kwarge["train"]
        self._tests=kwarge["test"]
        super().__init__(kwarge["Namefile"])
        self._result_x,self._result_y=super().GetValue

    @property
    def _train(self):
        train_x=self._result_x[:int(len(self._result_x) * self._traing)]
        train_y=self._result_y[:int(len(self._result_y) * self._traing)]
        
        return train_x,train_y

    @property
    def _test(self):
        test_x=self._result_x[-int(len(self._result_x) * self._tests):]
        test_y=self._result_y[-int(len(self._result_y) * self._tests):]
        return test_x,test_y
    
    def Load(self,batch_size,str_check):
        """
        Resizing and Converting image in ideal form to train
        n: Count Image
        Value: select Train or test
        return x,y
        """
        x_out = []
        y_out = []

        if str_check == "train":
            value=self._train
            n=self._Number_Train
        else:
            value=self._test
            n=self._Number_Test

        for i in range(0, batch_size):
            #Resizing and Converting image in ideal form to train
            x_out.append(scipy.misc.imresize(scipy.misc.imread(value[0][(self._batch_pointer + i) % n])[-150:], [66, 200]) / 255.0)
            y_out.append([value[1][(self._batch_pointer + i) % n]])
        train_batch_pointer += batch_size
        return x_out, y_out

    @property
    def _LenImge(self):
        return len(self._result_x)

    @property
    def _Number_Train(self):
        return self._train[0].__len__()
    
    @property
    def _Number_Test(self):
        return self._test[0].__len__()


        
if __name__ == '__main__':
    a=Train_Test(train=0.8,test=0.2,Namefile="../data.txt")
    print(a.Load(60,"test"))
        