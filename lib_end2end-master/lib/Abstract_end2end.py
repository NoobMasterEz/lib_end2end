
from tensorflow import nn ,Variable,truncated_normal,constant

class funtion_implement(object):
    @staticmethod
    def weight_variable(shape):
        return Variable(truncated_normal(shape, stddev=0.1))

    @staticmethod
    def bias_variable(shape):
        return Variable(constant(0.1,shape=shape))
    
    @staticmethod
    def conv2d(x, W, stride):\
        """
        input: A Tensor. Must be one of the following types: half, bfloat16, float32, float64. A 4-D tensor. 
                The dimension order is interpreted according to the value of data_format, see below for details.
        filters: A Tensor. Must have the same type as input. A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
        strides: An int or list of ints that has length 1, 2 or 4. 
                The stride of the sliding window for each dimension of input. If a single value is given it is replicated in the H and W dimension. 
                By default the N and C dimensions are set to 1. The dimension order is determined by the value of data_format, see below for details.
        padding: Either the string "SAME" or "VALID" indicating the type of padding algorithm to use, 
                or a list indicating the explicit paddings at the start and end of each dimension. 
                When explicit padding is used and data_format is "NHWC", 
                this should be in the form [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]. 
                When explicit padding used and data_format is "NCHW", 
                this should be in the form [[0, 0], [0, 0], [pad_top, pad_bottom], [pad_left, pad_right]].

        """
        return nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

class ReturnValue():
    y1:int
    y2:int
    y3:int 

class Model_e2e(funtion_implement):
    
    _INSTANCE_SHAPE=[-1, 1152]

    def __init__(self,X,Y):
        self.X,self.Y=X,Y

    @property
    def layer_conv(self,**kwage):

        W_conv = weight_variable(kwage["bias_variable"])
        b_conv = bias_variable(kwage["weight_variable"])
        h_conv = nn.relu(conv2d(kwage["x_image"], W_conv, kwage["stride"]) + b_conv)
        return W_conv, b_conv, h_conv

    @property
    def layer_FirstConnectNN(self,**kwage):
        
        W_fc1 = weight_variable(kwage["weight_variable"])
        b_fc1 = bias_variable(kwage["bias_variable"])

        h_conv5_flat = tf.reshape(kwage["h_conv5s"], _INSTANCE_SHAPE)


        keep_prob = tf.placeholder(tf.float32)
        h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        return {
            "h_fc1"=h_fc1,
            "h_fc1_drop"=h_fc1_drop,
            "keep_prob"=keep_prob
        }

    @property
    def layer_FullyConnect(self,**kwage):

        W_fc1 = weight_variable(kwage["bias_variable"])
        b_fc1 = bias_variable(kwage["weight_variable"])

        h_fc1 = tf.nn.relu(tf.matmul(kwage["h_fc_drop"], W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, kwage["keep_prob"])

        return {
            "W_fc":W_fc1,
            "b_fc":b_fc1,
            "h_fc":h_fc1,
            "h_fc_drop":h_fc1_drop
        }
    
    @property
    def layer_LastConnect(self,**kwage):
        W_fc1 = weight_variable(kwage["bias_variable"])
        b_fc1 = bias_variable(kwage["weight_variable"])

        y = tf.multiply(tf.atan(tf.matmul(kwage["h_fc_drop"], kwage["W_fc"]) +kwage["b_fc"]), 2) #scale the atan output
        return y 
    