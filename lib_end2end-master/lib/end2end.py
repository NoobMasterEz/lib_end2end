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
        ##############################  CONVORUTION  ##########################
        # Layer 1 convor
        W_conv1, b_conv1, h_conv1=layer_conv(bias_variable=[5, 5, 3, 24],weight_variable=[24],x_image=self._X,stride=2)
        # Layer 2 convor
        W_conv2, b_conv2, h_conv2=layer_conv(bias_variable=[5, 5, 24, 36],weight_variable=[36],x_image=h_conv1,stride=2)
        #Layer 3  convor 
        W_conv3, b_conv3, h_conv3=layer_conv(bias_variable=[5, 5, 36, 480],weight_variable=[48],x_image=h_conv2,stride=2)
        # Layer 4 convor
        W_conv4, b_conv4, h_conv4=layer_conv(bias_variable=[5, 5, 48, 64],weight_variable=[64],x_image=h_conv3,stride=1)
        # Layer 5 convor 
        W_conv5, b_conv5, h_conv5=layer_conv(bias_variable=[5, 5, 64, 64],weight_variable=[64],x_image=h_conv4,stride=1)
        
        ############################## FULLYCONNECT ###########################

        fully_data_result_1=layer_FirstConnectNN(weight_variable=[1152, 1164],bias_variable=[1164],h_conv5s=h_conv5)
        fully_data_result_2=layer_FullyConnect(weight_variable=[1164, 100],bias_variable=[100],h_conv5s=fully_data_result_1["h_fc1_drop"],keep_prob=fully_data_result_1["keep_prob"])
        fully_data_result_3=layer_FullyConnect(weight_variable=[100, 50],bias_variable=[50],h_conv5s=fully_data_result_2["h_fc1_drop"],keep_prob=fully_data_result_1["keep_prob"])
        fully_data_result_4=layer_FullyConnect(weight_variable=[50, 10],bias_variable=[10],h_conv5s=fully_data_result_3["h_fc1_drop"],keep_prob=fully_data_result_1["keep_prob"])
        output=layer_FullyConnect(weight_variable=[10, 1],bias_variable=[1],h_conv5s=fully_data_result_4["h_fc1_drop"],W_fc=fully_data_result_4["W_fc"],b_fc=fully_data_result_4["b_fc"])
        return output


if __name__ == "__main__":
    
    a=E2E(x=[tf.float32,[None,66,200,3] ],y=[tf.float32,[None,1] ])

    a.test
