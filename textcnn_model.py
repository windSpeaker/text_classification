#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np

class TextCNN():
    def __init__(self, embedding_size=100, filter_size=[3, 4, 5], max_len=50, num_classes=2, vocab_size=18000):
        self.x = tf.placeholder(tf.int32, shape=[None, max_len], name="x")
        self.y = tf.placeholder(tf.int32, shape=[None, num_classes], name="y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.embedding_init = tf.placeholder(tf.float32, shape=[None, embedding_size], name="embedding_init")

        with tf.name_scope("embedding"):
            embedding = tf.Variable(tf.truncated_normal(shape=[vocab_size+1, embedding_size], stddev=0.01), name="embedding", trainable=False)
            self.embed_init_op = tf.assign(embedding, self.embedding_init)
            # output shape: [batch, MAX_LENGTH, embedding_size]
            embedded = tf.nn.embedding_lookup(embedding, self.x, name="lookup")
            # output shape: [batch, MAX_LENGTH, embedding_size, 1]
            embedded = tf.expand_dims(embedded, -1)

        pooled_res = []
        for h in filter_size:
            with tf.name_scope("conv_%s"%(h)):
                k = tf.Variable(tf.truncated_normal(shape=[h, embedding_size, 1, 1], stddev=0.01), name="kernel_%s"%(h))
                # [batch, len - h, 1, 1]
                conv = tf.nn.conv2d(embedded, k, strides=[1,1,1,1], padding="VALID", name="conv_%s"%(h))
                conv_h = conv.get_shape()[1]
                pooled = tf.nn.max_pool(conv, ksize=[1, conv_h, 1, 1], strides=[1,1,1,1], padding="VALID", name="max_pool_%s"%(h))
                # [batch, 1, 1, 1]
                pooled_res.append(pooled)

        with tf.name_scope("dropout"):
            flatten = tf.reshape(tf.concat(pooled_res, 3), shape=(-1, len(filter_size)))
            dropout = tf.nn.dropout(flatten, keep_prob=self.keep_prob)

        with tf.name_scope("fc"):
            W = tf.Variable(tf.truncated_normal(shape=(len(filter_size), num_classes), stddev=0.01), name="w")
            b = tf.Variable(tf.constant(0.0, shape=[num_classes]), name="b")
            output = tf.nn.xw_plus_b(dropout, W, b)

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.y)

            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.loss = tf.reduce_mean(losses) + 0.01*l2_loss

        with tf.name_scope("optimizer"):
            learning_rate = 0.0001
            self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        with tf.name_scope("acc"):
            prediction = tf.argmax(output, axis=1)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(self.y, axis=1)), tf.float32))

    def run(self, train_data, train_label, test_data, test_label, embedd=None):
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("./log", sess.graph)
            sess.run(tf.global_variables_initializer())
            if embedd is not None:
                # print "do embedding init"
                sess.run([self.embed_init_op], feed_dict = {
                    self.embedding_init:embedd
                })

            for epoch in range(100):
                shuffle_indices = np.random.permutation(len(train_data))
                total_x = train_data[shuffle_indices]
                total_y = train_label[shuffle_indices]
                batch_size = 50
                num_batch = len(total_x)/batch_size
                for i in range(num_batch):
                    batch_x = total_x[i*batch_size:(i+1)*batch_size,:]
                    batch_y = total_y[i*batch_size:(i+1)*batch_size,:]
                    sess.run([self.trainer], feed_dict={
                        self.x : batch_x,
                        self.y: batch_y,
                        self.keep_prob: 0.5,
                    })
                test_acc, = sess.run([self.acc], feed_dict = {
                    self.x: test_data,
                    self.y: test_label,
                    self.keep_prob: 1,
                })
                print "Epochs: %s, acc: %s"%(epoch, test_acc)

if __name__ == "__main__":
    import process_data_mr
    train_x, train_y, test_x, test_y, embedding, vocab_size, embedding_size, max_len = process_data_mr.load_mr()
    print "vocab_size: ", vocab_size
    print "embedding_size: ", embedding_size
    print "max_len: ", max_len
    print "embedding_size: ", embedding.shape

    model = TextCNN(num_classes=2, max_len=max_len, vocab_size=vocab_size, embedding_size=embedding_size)
    # import process_sst1
    # train_x, train_y, test_x, test_y = process_sst1.fetch_data()
    # import process_data_mr
    # train_x, train_y, test_x, test_y = process_data_mr.load_mr()
    model.run(train_x, train_y, test_x, test_y, embedd=embedding)
    #model.run(train_x, train_y, test_x, test_y)

