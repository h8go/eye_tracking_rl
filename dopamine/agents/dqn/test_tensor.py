import tensorflow as tf
import numpy as np
#
# state = np.zeros((1,84, 84, 4))
#
# # Mask creation
# sigma_mask = 5
# mask = np.zeros((252,252))
# for x in range(252):
#     for y in range(252):
#         mask[x][y] = np.exp(-( ( (x-126)**2 + (y-126)**2 ) / ( 2.0 * sigma_mask**2 ) ) )
#
# mask_tensor = tf.Variable(tf.zeros(shape=[84, 84, 84, 84], dtype=tf.float32))
# print('shape', mask_tensor.shape)
# for i in range(84):
#     print(i)
#     for j in range(84):
#         mask_tensor[i, j, :, :].assign(mask[126-i:210-i, 126-j:210-j])
#
# print(mask_tensor.shape)

# WORKING !!
# t1 = tf.ones(shape=[84, 84, 84, 84], dtype=tf.float32)
# t2 = tf.ones(shape=[1, 1, 84, 84], dtype=tf.float32)
#
# x = tf.multiply(t1, t2)
# print(x)

# WORKING !!!
# p = np.arange(25)
# p = np.reshape(p, (1, 1, 5,5))
# print(p)
# P = tf.Variable(initial_value=p, validate_shape=True, shape=(1,1,5,5))
# print(P)
