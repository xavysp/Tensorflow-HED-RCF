
from os import path

from models.loss_functions import *
import tensorflow.contrib.layers as slim
from utilities.utls import read_numpy_file, make_dirs

class hed_net():

    def __init__(self, args):
        self.args =args
        if args.use_trained_weights:
            weights_fir = 'models'
            self.vgg16_weights = read_numpy_file(weights_fir,args.vgg16_param)
        self.input = tf.placeholder(tf.float32,[None,args.image_height, \
            args.image_width, args.n_channels],name='input')

        self.edgemaps = tf.placeholder(tf.float32,shape=[
            None, args.image_height, args.image_width,1], name = 'label')
        m = args.mean_pixel_values

        self.mean = tf.constant([m[2],m[1],m[0]], dtype=tf.float32, shape=[1,1,1,3],
                                name='img_mean')if not args.use_nir else \
            tf.constant([m[2], m[1], m[0], m[-1]], dtype=tf.float32, shape=[1, 1, 1, 3],
                        name='img_mean')

        self.define_model()

    def variable_on_cpu(self,name, shape, initializer):

        with tf.device('/cpu:0'):
            var = tf.get_variable(name=name,shape=shape,initializer=initializer,dtype=tf.float32)
        return var

    def variable_with_weight_decay(self, name, shape, stddev,wd=None):

        var = self.variable_on_cpu(name=name,shape=shape, \
                initializer=tf.truncated_normal_initializer(mean=0.,stddev=stddev))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def conv2d(self, input, output, k, s, name='',use_bias=True,use_trained=False,weight_decay=None):

        if use_trained:
            with tf.variable_scope(name):


                w = tf.constant(self.vgg16_weights[name+'_W'], dtype=tf.float32)
                conv = tf.nn.conv2d(input=input,filter=w,strides=[1,s,s,1],padding="SAME",
                                    name=name)
                if use_bias:
                    b = tf.constant(self.vgg16_weights[name + '_b'],dtype=tf.float32)
                    conv = tf.nn.bias_add(conv,b)
                return tf.nn.relu(conv)
        else:
            in_size = input.get_shape().as_list()[-1]
            w = self.variable_with_weight_decay(name=name+'_W',shape=[k,k,in_size,output],
                                                stddev=0.01, wd=weight_decay) # previous  stddev=0.1
            conv = tf.nn.conv2d(input=input,filter=w,strides=[1, s, s, 1],padding="SAME",
                                name=name)
            if use_bias:
                b = self.variable_on_cpu(name=name + '_B', shape=[output],
                                         initializer=tf.constant_initializer(0.0))
                conv = tf.nn.bias_add(conv,b)

        return conv

    def max_polling(self, input,name=''):

        max_pool = tf.nn.max_pool(value=input, ksize=[1,2,2,1],strides=[1,2,2,1],
                                  padding='SAME',name=name)
        return max_pool

    def side_output(self,input,weigth_decay=None,name=None):

        output = self.conv2d(input=input,output=1, k=1,s=1,name=name+'_conv', use_bias=False,
                             weight_decay=self.args.weight_decay)

        output = tf.image.resize_bilinear(images=output,size=[self.args.image_height, \
                    self.args.image_width])
        return output


    def define_model(self):
        with tf.variable_scope(self.args.model_name.lower()):

            use_tw = self.args.use_trained_weights  # if true use trained weights from VGG16

            #block 1
            self.conv1_1 = self.conv2d(input=self.input,output=64,k=3,s=1,name='conv1_1',use_trained=True)
            self.conv1_2 = self.conv2d(input=self.conv1_1,output=64,k=3,s=1,name='conv1_2', use_trained=True)
            self.output_1 = self.side_output(self.conv1_2,name='output1')
            self.max_pool1 = self.max_polling(self.conv1_2,name='maxpool1')

            # block 2
            self.conv2_1 = self.conv2d(input=self.max_pool1,output=128,k=3,s=1,name='conv2_1',use_trained=True)
            self.conv2_2 = self.conv2d(input=self.conv2_1,output=128,k=3,s=1,name='conv2_2',use_trained=True)
            self.output_2 = self.side_output(self.conv2_2, name='output2')
            self.max_pool2 = self.max_polling(self.conv2_2, name='maxpool2')

            # block 3
            self.conv3_1 = self.conv2d(input=self.max_pool2,output=256,k=3,s=1,name='conv3_1',use_trained=True)
            self.conv3_2 = self.conv2d(input=self.conv3_1, output=256, k=3, s=1, name='conv3_2',use_trained=True)
            self.conv3_3 = self.conv2d(input=self.conv3_2, output=256, k=3, s=1, name='conv3_3',use_trained=True)
            self.output_3 = self.side_output(self.conv3_3, name='output3')
            self.max_pool3 = self.max_polling(self.conv3_3, name='maxpool3')

            # block 4
            self.conv4_1 = self.conv2d(input=self.max_pool3, output=512, k=3, s=1, name='conv4_1',use_trained=True)
            self.conv4_2 = self.conv2d(input=self.conv4_1, output=512, k=3, s=1, name='conv4_2',use_trained=True)
            self.conv4_3 = self.conv2d(input=self.conv4_2, output=512, k=3, s=1, name='conv4_3',use_trained=True)
            self.output_4 = self.side_output(self.conv4_3, name='output4')
            self.max_pool4 = self.max_polling(self.conv4_3, name='maxpool4')

            # block 5
            self.conv5_1 = self.conv2d(input=self.max_pool4, output=512, k=3, s=1, name='conv5_1',use_trained=True)
            self.conv5_2 = self.conv2d(input=self.conv5_1, output=512, k=3, s=1, name='conv5_2',use_trained=True)
            self.conv5_3 = self.conv2d(input=self.conv5_2, output=512, k=3, s=1, name='conv5_3',use_trained=True)
            self.output_5 = self.side_output(self.conv5_3, name='output5')

            self.side_outputs = [self.output_1, self.output_2, self.output_3,
                                 self.output_4, self.output_5]
            side_outputs = tf.concat(self.side_outputs, axis=3)

            fw = self.variable_on_cpu(name='fuse',shape=[1,1,len(self.side_outputs),1],
                                      initializer=tf.constant_initializer(0.2))
            self.fuse = tf.nn.conv2d(side_outputs,filter=fw,strides=[1,1,1,1],padding='SAME',
                                      name='fuse')

            self.outputs = self.side_outputs+[self.fuse]

    # ***************************END MODEL DEFINITION **************************

    def test_model(self, session):
        self.predictions = []

        for idx, b in enumerate(self.outputs):
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            self.predictions.append(output)

    def train_model(self, session):

        self.predictions = []
        self.loss = 0

        print('Deep supervision application set to {}'.format(self.args.deep_supervision))

        for idx, b in enumerate(self.side_outputs):
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            cost = sigmoid_cross_entropy_balanced(b, self.edgemaps, name='cross_entropy{}'.format(idx))

            self.predictions.append(output)
            if self.args.deep_supervision:
                tf.add_to_collection('losses', (self.args.loss_weights * cost))
                self.loss += (self.args.loss_weights * cost)
                #deep_supervision
        self.fuse_output = tf.nn.sigmoid(self.fuse, name='fuse')
        fuse_cost = sigmoid_cross_entropy_balanced(self.fuse, self.edgemaps, name='cross_entropy_fuse')

        self.predictions.append(self.fuse_output)
        self.loss += (self.args.loss_weights * fuse_cost)

        # *************evaluation code
        tf.add_to_collection('losses',self.loss)
        self.all_losses = tf.add_n(tf.get_collection('losses'), name='all_losses')
        mean_loss = tf.train.ExponentialMovingAverage(0.9, name='avg')
        all_loss = tf.get_collection('losses')
        self.mean_average_op = mean_loss.apply(all_loss+[self.all_losses])
        # end ***************************

        pred = tf.cast(tf.greater(self.fuse_output, 0.5), tf.int32, name='predictions')
        error = tf.cast(tf.not_equal(pred, tf.cast(self.edgemaps, tf.int32)), tf.float32)
        self.error = tf.reduce_mean(error, name='pixel_error')

        tf.summary.scalar('Train', self.all_losses) # previously self.loss
        tf.summary.scalar('Validation', self.error)

        train_log_dir = path.join(self.args.logs_dir, self.args.model_name.lower() +
                                  '_' + self.args.dataset4training.lower(),self.args.model_state)
        val_log_dir = path.join(self.args.logs_dir, self.args.model_name.lower() +
                                '_' + self.args.dataset4training.lower(), 'val')
        _ = make_dirs(train_log_dir)
        _ = make_dirs(val_log_dir)

        self.merged_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(train_log_dir, session.graph)
        self.val_writer = tf.summary.FileWriter(val_log_dir)