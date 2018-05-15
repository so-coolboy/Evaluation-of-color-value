# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:05:19 2018

@author: www
"""

from PIL import Image
import os
import tensorflow as tf
import numpy as np



#数据集地址
path = r'E:\face_score\cut_image'

#读取图片
def read_img(path):
     imgs = []
     labels = []
     for image in os.listdir(path):
          id_tag = image.find("-")
          score = image[0:id_tag]
          print('score',score)
          img = Image.open(os.path.join(path, image))
          img_ndarray = np.asarray(img, dtype='float32')
          img_ndarray = np.reshape(img_ndarray, [128, 128, 3])
          imgs.append(img_ndarray)
          score = (int(score)+5)/10
          label = np.asarray([0] * 10)
          label[int(score) - 1] = 1
          labels.append(label)
     return np.asarray(imgs,np.float32),np.asarray(labels,np.float32)
     
data,labels = read_img(path)



#模型         
dropout = 0.75
x = tf.placeholder(tf.float32, [None, 128, 128, 3])
y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)
     
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

    
def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 128, 128, 3])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print(conv1.shape)

    conv1 = maxpool2d(conv1, k=2)
    print(conv1.shape)
   
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print(conv2.shape)
    
    conv2 = maxpool2d(conv2, k=2)
    print(conv2.shape)
    
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
 
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    print(out.shape)
    return out
    
weights = {
    # 5x5 conv, 3 input, 24 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 24]),dtype=tf.float32),
    # 5x5 conv, 24 inputs, 96 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 24, 96]),dtype=tf.float32),
    # fully connected, 32*32*96 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([32*32*96, 1024]),dtype=tf.float32),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, 10]),dtype=tf.float32)
}

biases = {
    'bc1': tf.Variable(tf.random_normal([24]),dtype=tf.float32),
    'bc2': tf.Variable(tf.random_normal([96]),dtype=tf.float32),
    'bd1': tf.Variable(tf.random_normal([1024]),dtype=tf.float32),
    'out': tf.Variable(tf.random_normal([10]),dtype=tf.float32)
}    

pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1),name='output')
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver=tf.train.Saver()



#开始训练


batch_size = 32

sess = tf.Session()
sess.run(init)

train_loss= []
train_acc = []
for i in range(50):
     rand_index = np.random.choice(len(data), size=batch_size)
     rand_x = data[rand_index]
     rand_y = labels[rand_index]
     train_dict ={x:rand_x, y:rand_y, keep_prob: dropout}
     sess.run(optimizer,feed_dict=train_dict)
     print('done')
     if i % 10 == 0:
          loss, acc = sess.run([cost, accuracy], feed_dict={x: rand_x,y: rand_y,keep_prob: 1.})
          print("count = " + str(i) + ", Minibatch Loss= " + str(loss) + ", Training Accuracy= " + str(acc) )
print("Optimization Finished!")
output_graph_def  = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])  
with tf.gfile.FastGFile('test.pb', 'wb') as f:
     f.write(output_graph_def .SerializeToString())





