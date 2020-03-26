from lib.end2end import E2E
import tensorflow as tf

if __name__ == "__main__":
        a=E2E(
                x=[tf.float32,[None,66,200,3] ],
                y=[tf.float32,[None,1] ],
                batch_size=100,
                epochs=60)

        a.fit()

