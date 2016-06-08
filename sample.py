import numpy as np
import tensorflow as tf

import os
import sys

from reader import getDicts
from reader import read_poems
from train import trainModel

checkpoint_dir = os.path.join('.')
exclusion = ['*']

print 'Character: ', sys.argv[1]

vocab_size = 2000

index_to_char, char_to_index = getDicts(vocab_size)
data = read_poems(char_to_index)
with tf.variable_scope("trainModel"):
    model = trainModel(training=False, infer=True)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    print '-------', type(tf.all_variables())
    for a in tf.all_variables():
        print a.name
    print '-------'

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    print 'ckpt.model_checkpoint_path: ', ckpt.model_checkpoint_path

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print model.sample(sess, index_to_char, char_to_index, sys.argv[1])
    else:
        print 'else, no data to restore'

