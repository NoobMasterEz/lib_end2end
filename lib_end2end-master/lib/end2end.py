import tensorflow.compat.v1 as tf
import Driving
import os 
from .Abstract_end2end import Model_e2e as fi 
from tensorflow.core.protobuf import saver_pb2
tf.disable_v2_behavior()





class E2E(fi):

    """
    Ex.
    if __name__ == "__main__":
    
        a=E2E(x=[tf.float32,[None,66,200,3] ],y=[tf.float32,[None,1] ])

        a.test

    """

    _X=None
    _Y=None
    L2NORMCONSTAND=0.001
    LOGDIR='./save'
    LOGS_PATH = './logs'
        
    
    
    def __init__(self,*args, **kwargs):
        """

        Model Test driveing 
        X: set is placeholder to  input defut type float 32 , shape=[None,66,200,3]
        Y: set is placeholder to  input defut type float 32 , shape=[None,1]  
        """
        self._X= tf.placeholder(kwargs["x"][0],kwargs["x"][1])
        self._Y= tf.placeholder(kwargs["y"][0],kwargs["y"][1])
        self.Batch_size=kwargs["batch_size"]
        self.Epochs=kwargs["epochs"]
        self.Session=tf.InteractiveSession()
        super().__init__(self._X,self._Y)

    @property
    def Model(self):
        ##############################  CONVORUTION  ##########################
        # Layer 1 convor 24@31x98
        W_conv1, b_conv1, h_conv1=fi.layer_conv(bias_variable=[5, 5, 3, 24],weight_variable=[24],x_image=self._X,stride=2)
        # Layer 2 convor 36@14x47
        W_conv2, b_conv2, h_conv2=fi.layer_conv(bias_variable=[5, 5, 24, 36],weight_variable=[36],x_image=h_conv1,stride=2)
        #Layer 3  convor 48@5x22
        W_conv3, b_conv3, h_conv3=fi.layer_conv(bias_variable=[5, 5, 36, 48],weight_variable=[48],x_image=h_conv2,stride=2)
        # Layer 4 convor 64@3x20
        W_conv4, b_conv4, h_conv4=fi.layer_conv(bias_variable=[3, 3, 48, 64],weight_variable=[64],x_image=h_conv3,stride=1)
        # Layer 5 convor 64@1x18
        W_conv5, b_conv5, h_conv5=fi.layer_conv(bias_variable=[3, 3, 64, 64],weight_variable=[64],x_image=h_conv4,stride=1)
        
        ############################## FULLYCONNECT ###########################
        """
        Nvidia-end-to-end-self-driving-cars

        |_1164@1x1
        |_100@1x1
        |_50@1x1
        |_10@1x1
        |_1@1x1

        """
        self.fully_data_result_1=fi.layer_FirstConnectNN(weight_variable=[1152, 1164],bias_variable=[1164],h_conv5s=h_conv5)
        fully_data_result_2=fi.layer_FullyConnect(weight_variable=[1164, 100],bias_variable=[100],h_fc_drop=self.fully_data_result_1["h_fc1_drop"],keep_prob=self.fully_data_result_1["keep_prob"])
        fully_data_result_3=fi.layer_FullyConnect(weight_variable=[100, 50],bias_variable=[50],h_conv5s=fully_data_result_2["h_fc1_drop"],keep_prob=self.fully_data_result_1["keep_prob"])
        fully_data_result_4=fi.layer_FullyConnect(weight_variable=[50, 10],bias_variable=[10],h_conv5s=fully_data_result_3["h_fc1_drop"],keep_prob=self.fully_data_result_1["keep_prob"])
        output=fi.layer_LastConnect(weight_variable=[10, 1],bias_variable=[1],h_conv5s=fully_data_result_4["h_fc1_drop"],W_fc=fully_data_result_4["W_fc"],b_fc=fully_data_result_4["b_fc"])
        return output

    @property
    def __Loss(self):
        return tf.reduce_mean(tf.square(tf.subtract(self._Y,self.Model())) + tf.add_n([tf.nn.l2_loss(i) for i in self.Session]))*self.L2NORMCONSTAND

    @property
    def __train_stepByAdam(self):
        return tf.train.AdamOptimizer(1e-4).minimize(self.__Loss)

    @property
    def Board(self):
        # create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.__Loss)
        # merge all summaries into a single op
        return tf.summary.merge_all()
        


    def fit(self):
        
        self.Session.run(tf.initialize_all_variables())
        merged_summary_op=self.Board()
        saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)
        # op to write logs to Tensorboard
        train,val =Driving.Driving_Data(self.Batch_size).Getter_load_data()
        summary_writer = tf.summary.FileWriter(self.LOGS_PATH, graph=tf.get_default_graph())
        for epoch in range(self.Epochs):
            for i in range(int(Driving.Driving_Data.Number_image/self.Batch_size)):
                # Loade train data
                x,y=train
                self.__train_stepByAdam.run(feed_dict={self._X:x,self._Y:y,self.fully_data_result_1:0.8})
                if i%10== 0:
                    x,y=val
                    loss_value = self.__Loss.eval(feed_dict={self._X:x,self._Y:y,self.fully_data_result_1:1.0})
                    print("Epoch: %d, Step: %d, Loss: %g" % (self.Epochs, self.Epochs * self.Batch_size + i, loss_value))
                
                # write logs at every iteration
                summary = merged_summary_op.eval(feed_dict={self._X:x,self._Y:y,self.fully_data_result_1:1.0})
                summary_writer.add_summary(summary, self.Epochs * Driving.Driving_Data.Number_image /self.Batch_size + i)

                if i ==0 :
                    if not os.path.exists(self.LOGDIR):
                        os.makedirs(self.LOGDIR)
                    checkpoint_path = os.path.join(self.LOGDIR, "model.ckpt")
                    filename = saver.save(self.Session, checkpoint_path)
                print("Model saved in file: %s" % filename)
        print("Run the command line:\n" \
                "--> tensorboard --logdir=./logs " \
                "\nThen open http://0.0.0.0:6006/ into your web browser")