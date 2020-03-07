import tensorflow as tf 
from Abstract_end2end import funtion_implement as fi



class Model(object):
    _X=None
    _Y=None

    def __init__(self,*args, **kwargs):
        """
        Model Test driveing 
        X: set is placeholder to  input defut type float 32 , shape=[None,66,200,3]
        Y: set is placeholder to  input defut type float 32 , shape=[None,1]  
        """
        self._X= tf.placeholder(kwargs["x"][0],kwargs["x"][1])
        self._Y= tf.placeholder(kwargs["y"][0],kwargs["y"][1])
    
    @property
    def test(self):
        print(fi.weight_variable([5, 5, 3, 24]))


if __name__ == "__main__":
    
    a=Model(x=[tf.float32,[None,66,200,3] ],y=[tf.float32,[None,1] ])

    a.test
