# from tensorflow.keras import backend as K
# import tensorflow as tf
# y_true = tf.constant([[0, 0], [0, 0]], dtype=tf.float32)
# y_pred = tf.constant([[0, 0], [0, 1]], dtype=tf.float32)
# loss =  (K.binary_crossentropy(y_true, y_pred))
# print(K.mean(loss))
# print((K.mean(loss[0])+K.mean(loss[1]))/2)

# # assert loss.shape == (2,)
# # loss.numpy()

bobloss = 4
print(bobloss)
eveloss = 0.7

loss = bobloss - 7 * min(eveloss, 0.5)

print(loss)