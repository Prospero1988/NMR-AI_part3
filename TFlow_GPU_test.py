import tensorflow as tf

print("TensorFlow version:", tf.__version__)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPUs: ", tf.config.list_physical_devices('GPU'))

# Simple test operation
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0]])
        b = tf.constant([[1.0], [2.0], [3.0]])
        c = tf.matmul(a, b)
        print("Result of matrix multiplication:", c)
else:
    print("No GPUs found.")
