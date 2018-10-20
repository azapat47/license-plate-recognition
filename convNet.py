import os
import time 
import tensorflow as tf
import tools

#Hyperparams
TOTAL=560
TESTS=10
TRAIN=10
PERIODS=TRAIN

def conv_relu():
    pass

def maxpool():
    pass

def fully_connected():
    pass

class ConvNetwork():
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 128
        #self.keep_prob = tf.constant(0.75)
        self.globalstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        self.n_classes = 10+26
        self.skip_step = 20
        self.n_test = TESTS
        self.training = True
        # Get data
        with tf.name_scope('data'):
            train_data, test_data = tools.get_dataset(self.batch_size)
            iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                       train_data.output_shapes)
            img, self.label = iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])
            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)    # initializer for train_data


    def _loss(self):
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')

    def _optimize(self):
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=self.globalstep)

    def _create_summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    # N of Right predictions
    def _eval_model(self):
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build_graph(self):
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.create_summary()

    def _train_epoch(self, sess, saver, init, writer, epoch, step):
        print(sess)
        print(saver)
        print(init)
        print(writer)
        print(epoch)
        print(step)
        pass

    def _eval_once(self):
        pass
    
    def train(self, periods,checkpoint_frequency):
        tools.makedir('checkpoints/convnet')
        writer = tf.summary.FileWriter('./graphs/convnet', tf.get_default_graph())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            checkpoint = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet/checkpoint'))
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
            step = self.globalstep.eval()
            for epoch in range(periods):
                step = self.train_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
                if (epoch + 1) % checkpoint_frequency == 0:
                    saver.save(sess, 'checkpoints/convnet/checkpoint', index)
        writer.close()

if __name__ == '__main__':
    model = ConvNetwork()
    model.build_graph()
    model.train(periods=PERIODS)
