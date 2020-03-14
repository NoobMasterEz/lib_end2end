
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


class Model_e2e(funtion_implement):


    def __init__(self,X,Y):
        self.X,self.Y=X,Y

    @property
    def layer_conv(self,*arges,**kwage):

        W_conv = weight_variable(arges[0])
        b_conv = bias_variable(arges[1])
        h_conv = nn.relu(conv2d(kwage["x_image"], W_conv, kwage["stride"]) + b_conv)

        return W_conv, b_conv, h_conv

    