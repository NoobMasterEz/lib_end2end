import tensorflow.compat.v1 as tf
from Abstract_end2end import funtion_implement as fi
from Abstract_end2end import Model_e2e
tf.compat.v1.disable_eager_execution()

__metaclass__ = type

class E2E(Model_e2e):
    _X,_Y=None,None
    _STRIDE=2
    LOGDIR='./save'
    def __init__(self,*args, **kwargs):
        """
        Model Test driveing 
        X: set is placeholder to  input defut type float 32 , shape=[None,66,200,3]
        Y: set is placeholder to  input defut type float 32 , shape=[None,1]  
        """
        self._X= tf.placeholder(kwargs["x"][0],kwargs["x"][1])
        self._Y= tf.placeholder(kwargs["y"][0],kwargs["y"][1])
        super(E2E,self).__init__(self._X,self._Y)

        #defind session interractive
        self.sess = tf.InteractiveSession()

    def getter_x(self):
        return self._X

    @property
    def Model(self):
        """
        self._STRIDE defualt to 2 *2
        """
        # layer 1 
        W_conv1, b_conv1, h_conv1=Model_e2e.layer_conv(kernel=[5, 5, 3, 24],shape=[24],x_image=self.getter_x ,stride=self._STRIDE)
        # layer 2 
        W_conv2, b_conv2, h_conv2=Model_e2e.layer_conv(kernel=[5, 5, 24, 36],shape=[36],x_image=h_conv1 ,stride=self._STRIDE)
        # layer 3
        W_conv2, b_conv2, h_conv2=Model_e2e.layer_conv(kernel=[5, 5, 36, 48],shape=[48],x_image=h_conv2 ,stride=self._STRIDE)

    def train

if __name__ == "__main__":
    
    a=E2E(x=[tf.float32,[None,66,200,3] ],y=[tf.float32,[None,1] ])

    a.Model
