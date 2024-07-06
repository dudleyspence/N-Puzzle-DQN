import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check for available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {len(gpus)}")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPUs detected")

# Perform a simple computation on the GPU if available
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = tf.matmul(a, b, transpose_b=True)
    print("Matrix multiplication result:\n", c)
