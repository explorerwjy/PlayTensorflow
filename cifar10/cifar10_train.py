# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

init_lr = 1e-4
optimizer = 'Adam'

class LossQueue:
    def __init__(self):
        self.queue = deque([500] * 100, 100)
    def enqueue(self, value):
        self.queue.appendleft(value)
        self.queue.pop()
    def avgloss(self):
        res = list(self.queue)
        return float(sum(res))/len(res)

def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        images, labels = cifar10.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        resnet = cifar10.ResNet()
        logits = resnet.Inference(images)

        # Calculate loss.
        loss = resnet.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = resnet.train(loss, global_step, init_lr, optimizer)

        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        sess.run(init)
        print "Before Queue"
        # Start the queue runners.
        coord = tf.train.Coordinator()

        print "Before Threads"
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        loss_queue = LossQueue() 
        min_loss = loss_queue.avgloss()
        try:    
            print "Start"
            if continueModel != None:
                saver.restore(sess, continueModel)
                print "Continue Train Mode. Start with step",sess.run(global_step) 
                
            for step in xrange(FLAGS.max_steps):
                if coord.should_stop():
                    break
                start_time = time.time()

                _, loss_value, v_step = sess.run([train_op, loss, global_step])
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if v_step % 10 == 0:
                    loss_queue.enqueue(loss_value)
                    #avgloss = loss_queue.avgloss()
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / FLAGS.num_gpus
                    format_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f '
                      'sec/batch)')
                    print (format_str % (datetime.now(), v_step, loss_value,
                             examples_per_sec, sec_per_batch))
                
                if v_step % 100 == 0:
                    prediction = float((np.sum(sess.run(top_k_op))))
                    print '@ Step {}: \t {} in {} Correct, Batch precision @ 1 ={}'.format(v_step, prediction, self.batch_size, prediction/self.batch_size)
                    #print accuracy
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, v_step)

                # Save the model checkpoint periodically.
                if v_step % 1000 == 0 or (v_step + 1) == FLAGS.max_steps:
                    #self.EvalWhileTraining()
                    avgloss = loss_queue.avgloss()
                    if avgloss < min_loss:
                        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=v_step)
                        min_loss = avgloss 
                        print "Write A CheckPoint at %d with avgloss %.5f" % (v_step, min_loss)
                    else:
                        print "Current Min avgloss is %.5f. Last avgloss is %.5f" % ( min_loss, avgloss)
        except Exception, e:
            coord.request_stop(e)
        finally:
            sess.run(queue.close(cancel_pending_enqueues=True))
            coord.request_stop()
            coord.join()

def main(argv=None):  # pylint: disable=unused-argument
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
