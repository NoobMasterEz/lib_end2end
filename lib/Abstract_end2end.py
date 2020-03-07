from tensorflow import nn ,Variable,truncated_normal,constant 

class funtion_implement(object):
    @staticmethod
    def weight_variable(shape):
        return Variable(truncated_normal(shape, stddev=0.1))

    @staticmethod
    def bias_variable(shape):
        raise NotImplementedError
    
    @staticmethod
    def conv2d(shape):
        raise NotImplementedError