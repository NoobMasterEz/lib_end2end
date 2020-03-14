import tensorflow as tf 
from Abstract_end2end import funtion_implement as fi



class E2E(fi):
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
        super(self._X,self._Y)

    @property
    def Model(self):
        W_conv, b_conv, h_conv=layer_conv([5, 5, 3, 24],[24],x_image=self._X)
        print(W_conv, b_conv, h_conv)


if __name__ == "__main__":
    
    a=E2E(x=[tf.float32,[None,66,200,3] ],y=[tf.float32,[None,1] ])

    a.test
