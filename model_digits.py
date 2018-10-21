import os
import time 
import tensorflow as tf
import tools
import numpy as np

#Hyperparams
TOTAL=560
TESTS=10
TRAIN=10
PERIODS=10
FOLDER="data/digitos/"

def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_channels = inputs.shape[-1]
        kernel = tf.get_variable('kernel', 
                                 [k_size, k_size, in_channels, filters], 
                                 initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', 
                                 [filters],
                                 initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.relu(conv + biases, name=scope.name)

def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs, 
                              ksize=[1, ksize, ksize, 1], 
                              strides=[1, stride, stride, 1],
                              padding=padding)
    return pool

def fully_connected(inputs, out_dim, scope_name='fc'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable('weights', [in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b
    return out

class ConvNetwork(object):
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 25
        #self.keep_prob = tf.constant(0.75)
        self.keep_prob = tf.constant(0.5)
        self.globalstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        self.n_classes = 10
        self.skip_step = 20
        self.n_test = TESTS
        self.training = True
        
    def get_data(self):
        with tf.name_scope('data'):
            folder = FOLDER
            train_data, test_data = tools.get_dataset(folder,TRAIN,TESTS,self.batch_size, self.n_classes)
            self.iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                       train_data.output_shapes)
            img, self.label = self.iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])
            self.train_init = self.iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = self.iterator.make_initializer(test_data)    # initializer for train_data
            #self.img_pred = tf.placeholder(shape=[1, 28, 28, 1], dtype=tf.float32)
            
    def inference(self):
        conv1 = conv_relu(inputs=self.img,
                        filters=32,
                        k_size=5,
                        stride=1,
                        padding='SAME',
                        scope_name='conv1')
        pool1 = maxpool(conv1, 2, 2, 'VALID', 'pool1')
        conv2 = conv_relu(inputs=pool1,
                        filters=64,
                        k_size=5,
                        stride=1,
                        padding='SAME',
                        scope_name='conv2')
        pool2 = maxpool(conv2, 2, 2, 'VALID', 'pool2')
        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, feature_dim])
        fc = fully_connected(pool2, 1024, 'fc')
        dropout = tf.nn.dropout(tf.nn.relu(fc), self.keep_prob, name='relu_dropout')
        self.logits = fully_connected(dropout, self.n_classes, 'logits')
            
    def loss(self):
        with tf.name_scope('loss'):
            self.entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(self.entropy, name='loss')

    def optimize(self):
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=self.globalstep)

    def create_summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    # N of Right predictions
    def eval_model(self):
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build_graph(self):
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval_model()
        self.create_summary()

    def train_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init) 
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                #print(self.logits.eval().shape, 'logits')
                #print(self.label.eval().shape, 'labels')
                #print(np.argmax(self.logits.eval(), axis=1))
                #print(np.argmax(self.label.eval(), axis=1))
                #print(self.entropy.eval())
                #print(self.loss.eval())
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'checkpoints/convnet_digits/digits-convnet', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/self.n_test))
        print('global step is: {0}'.format(step))
        print('Took: {0} seconds'.format(time.time() - start_time), end='\n\n\n')
    
    def train(self, periods,checkpoint_frequency):
        tools.makedir('checkpoints/convnet')
        writer = tf.summary.FileWriter('./graphs/convnet-digits', tf.get_default_graph())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # checkpoint = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet/checkpoint'))
            checkpoint = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_digits/checkpoint'))
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
            step = self.globalstep.eval()
            for epoch in range(step, periods):
                step = self.train_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
                if (epoch + 1) % checkpoint_frequency == 0:
                    saver.save(sess, 'checkpoints/convnet_digits/checkpoint', epoch)
        writer.close()

    def predict(self, img, label):
        img = tf.reshape(img, shape=[1, 28, 28])
        label = tf.reshape(label, shape=[1, 10])
        self.training = False
        
        pred_dataset = tf.data.Dataset.from_tensor_slices((img, label))
        pred_data = pred_dataset.batch(self.batch_size)
        
        self.pred_init = self.iterator.make_initializer(pred_data)
        out = tf.nn.softmax(self.logits)
        clase = tf.argmax(out, 1)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.pred_init)
            pred = sess.run(clase)
        
        return pred
        
if __name__ == '__main__':
    model = ConvNetwork()
    model.build_graph()
    model.train(PERIODS,20)
    img, label = tools.loadImage(FOLDER, '0-0-coplate34.png', 10)
    pred = model.predict(img, label)
    print(pred)
