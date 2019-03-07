
from os import path
import numpy as np

from models.loss_functions import *
from utilities.utls import read_pretrained_data, make_dirs

class hed_net():

    def __init__(self, args):
        self.args =args
        self.im_height= args.image_height
        self.im_width = args.image_width
        if args.use_trained_weights:
            weights_fir = 'models'
            self.vgg16_weights = read_pretrained_data(weights_fir,args.vgg16_param)
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
                initializer=tf.keras.initializers.Zeros())  # previous tf.truncated_normal_initializer(mean=0.,stdded=0.01)
        # tf.contrib.layers.xavier_initializer() / tf.keras.initializers.Zeros()
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name=name+'_weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def get_bilinear_weight(self, k_size,out_chanels=1,name=''):

        factor = (k_size+1)//2
        if k_size%2==1:
            center =factor-1
        else:
            center=factor-0.5
        og = np.ogrid[:k_size,:k_size]
        filt = (1-abs(og[0]-center)/factor)*(1-abs(og[1]-center)/factor)
        w=np.zeros((k_size,k_size,out_chanels,out_chanels))
        for i in range(out_chanels):
            for j in range(out_chanels):
                if i==j:
                    w[:,:,i,j]=filt
        init = tf.constant_initializer(value=w,dtype=tf.float32)
        return tf.get_variable(name=name, initializer=init,shape=w.shape)

    def conv2d(self, input, output, k, s, name='',use_bias=True,use_trained=False,weight_decay=None):

        if use_trained:
            with tf.variable_scope(name):

                w_shape = self.vgg16_weights[name+'_W'].shape
                w_init = tf.constant_initializer(self.vgg16_weights[name+'_W'], dtype=tf.float32)
                w = tf.get_variable(name=name+'_W', initializer=w_init, shape=w_shape)
                conv = tf.nn.conv2d(input=input,filter=w,strides=[1,s,s,1],padding="SAME",
                                    name=name)
                if use_bias:
                    b_shape = self.vgg16_weights[name + '_b'].shape
                    b_init =tf.constant_initializer(self.vgg16_weights[name + '_b'],dtype=tf.float32)
                    b = tf.get_variable(name=name+'_b',shape=b_shape, initializer=b_init)
                    conv = tf.nn.bias_add(conv,b)
                return tf.nn.relu(conv)
        else:
            in_size = input.get_shape().as_list()[-1]
            w = self.variable_with_weight_decay(name=name+'_W',shape=[k,k,in_size,output],
                                                stddev=0.01, wd=weight_decay) # previous  stddev=0.1
            conv = tf.nn.conv2d(input=input,filter=w,strides=[1, s, s, 1],padding="SAME",
                                name=name)
            if use_bias:
                b = self.variable_on_cpu(name=name + '_b', shape=[output],
                                         initializer=tf.constant_initializer(0.0))
                conv = tf.nn.bias_add(conv,b)

        return conv

    def max_polling(self, input,name=''):

        max_pool = tf.nn.max_pool(value=input, ksize=[1,2,2,1],strides=[1,2,2,1],
                                  padding='SAME',name=name)
        return max_pool

    def _upscore_layer(self, input, shape, n_outputs, name,
                       ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = input.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(input)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, n_outputs]
            else:
                new_shape = [shape[0], shape[1], shape[2], n_outputs]
            output_shape = tf.stack(new_shape)

            f_shape = [ksize, ksize, n_outputs, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input) ** 0.5

            weights = self.get_deconv_filter(f_shape)
            deconv = tf.nn.conv2d_transpose(input, weights, output_shape,
                                            strides=strides, padding='SAME')
        # _activation_summary(deconv)
        return deconv

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        heigh = f_shape[0]
        f = np.ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)

        return tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)

    def side_output(self,input,weigth_decay=None,name=None, up_scale=2):

        output = self.conv2d(input=input,output=1, k=1,s=1,name=name+'_conv', use_bias=False,
                             weight_decay=self.args.weight_decay)
        c_shape = output.get_shape().as_list()[2]

        if c_shape != self.args.image_width:
            output = tf.image.resize_bilinear(images=output,size=[self.args.image_height, \
                        self.args.image_width],align_corners=True)
            # in_shape = tf.shape(input)
            # out_shape = [in_shape[0],self.args.image_height,self.args.image_width,1]
            # output = self.up_sampling(input=output,stride=2,output_shape=out_shape,
            #                           name=name+'_ldconv',up_scale=up_scale)
        return output

    def prepare_aligned_crop(self):
        """ Prepare for aligned crop. """
        # Re-implement the logic in deploy.prototxt and
        #   /hed/src/caffe/layers/crop_layer.cpp of official repo.
        # Other reference materials:
        #   hed/include/caffe/layer.hpp
        #   hed/include/caffe/vision_layers.hpp
        #   hed/include/caffe/util/coords.hpp
        #   https://groups.google.com/forum/#!topic/caffe-users/YSRYy7Nd9J8

        def map_inv(m):
            """ Mapping inverse. """
            a, b = m
            return 1 / a, -b / a

        def map_compose(m1, m2):
            """ Mapping compose. """
            a1, b1 = m1
            a2, b2 = m2
            return a1 * a2, a1 * b2 + b1

        def deconv_map(kernel_h, stride_h, pad_h):
            """ Deconvolution coordinates mapping. """
            return stride_h, (kernel_h - 1) / 2 - pad_h

        def conv_map(kernel_h, stride_h, pad_h):
            """ Convolution coordinates mapping. """
            return map_inv(deconv_map(kernel_h, stride_h, pad_h))

        def pool_map(kernel_h, stride_h, pad_h):
            """ Pooling coordinates mapping. """
            return conv_map(kernel_h, stride_h, pad_h)

        x_map = (1, 0)
        conv1_1_map = map_compose(conv_map(3, 1, 35), x_map)
        conv1_2_map = map_compose(conv_map(3, 1, 1), conv1_1_map)
        pool1_map = map_compose(pool_map(2, 2, 0), conv1_2_map)

        conv2_1_map = map_compose(conv_map(3, 1, 1), pool1_map)
        conv2_2_map = map_compose(conv_map(3, 1, 1), conv2_1_map)
        pool2_map = map_compose(pool_map(2, 2, 0), conv2_2_map)

        conv3_1_map = map_compose(conv_map(3, 1, 1), pool2_map)
        conv3_2_map = map_compose(conv_map(3, 1, 1), conv3_1_map)
        conv3_3_map = map_compose(conv_map(3, 1, 1), conv3_2_map)
        pool3_map = map_compose(pool_map(2, 2, 0), conv3_3_map)

        conv4_1_map = map_compose(conv_map(3, 1, 1), pool3_map)
        conv4_2_map = map_compose(conv_map(3, 1, 1), conv4_1_map)
        conv4_3_map = map_compose(conv_map(3, 1, 1), conv4_2_map)
        pool4_map = map_compose(pool_map(2, 2, 0), conv4_3_map)

        conv5_1_map = map_compose(conv_map(3, 1, 1), pool4_map)
        conv5_2_map = map_compose(conv_map(3, 1, 1), conv5_1_map)
        conv5_3_map = map_compose(conv_map(3, 1, 1), conv5_2_map)

        score_dsn1_map = conv1_2_map
        score_dsn2_map = conv2_2_map
        score_dsn3_map = conv3_3_map
        score_dsn4_map = conv4_3_map
        score_dsn5_map = conv5_3_map

        upsample2_map = map_compose(deconv_map(4, 2, 0), score_dsn2_map)
        upsample3_map = map_compose(deconv_map(8, 4, 0), score_dsn3_map)
        upsample4_map = map_compose(deconv_map(16, 8, 0), score_dsn4_map)
        upsample5_map = map_compose(deconv_map(32, 16, 0), score_dsn5_map)

        crop1_margin = int(score_dsn1_map[1])
        crop2_margin = int(upsample2_map[1])
        crop3_margin = int(upsample3_map[1])
        crop4_margin = int(upsample4_map[1])
        crop5_margin = int(upsample5_map[1])

        return crop1_margin, crop2_margin, crop3_margin, crop4_margin, crop5_margin

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
            self.output_2 = self.side_output(self.conv2_2, name='output2', up_scale=2)
            self.max_pool2 = self.max_polling(self.conv2_2, name='maxpool2')

            # block 3
            self.conv3_1 = self.conv2d(input=self.max_pool2,output=256,k=3,s=1,name='conv3_1',use_trained=True)
            self.conv3_2 = self.conv2d(input=self.conv3_1, output=256, k=3, s=1, name='conv3_2',use_trained=True)
            self.conv3_3 = self.conv2d(input=self.conv3_2, output=256, k=3, s=1, name='conv3_3',use_trained=True)
            self.output_3 = self.side_output(self.conv3_3, name='output3', up_scale=4)
            self.max_pool3 = self.max_polling(self.conv3_3, name='maxpool3')

            # block 4
            self.conv4_1 = self.conv2d(input=self.max_pool3, output=512, k=3, s=1, name='conv4_1',use_trained=True)
            self.conv4_2 = self.conv2d(input=self.conv4_1, output=512, k=3, s=1, name='conv4_2',use_trained=True)
            self.conv4_3 = self.conv2d(input=self.conv4_2, output=512, k=3, s=1, name='conv4_3',use_trained=True)
            self.output_4 = self.side_output(self.conv4_3, name='output4',up_scale=8)
            self.max_pool4 = self.max_polling(self.conv4_3, name='maxpool4')

            # block 5
            self.conv5_1 = self.conv2d(input=self.max_pool4, output=512, k=3, s=1, name='conv5_1',use_trained=True)
            self.conv5_2 = self.conv2d(input=self.conv5_1, output=512, k=3, s=1, name='conv5_2',use_trained=True)
            self.conv5_3 = self.conv2d(input=self.conv5_2, output=512, k=3, s=1, name='conv5_3',use_trained=True)
            self.output_5 = self.side_output(self.conv5_3, name='output5',up_scale=16)

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
            # before sigmoid_cross_entropy_balanced

            self.predictions.append(output)
            if self.args.deep_supervision:
                s_cost = (self.args.loss_weights * cost)
                tf.add_to_collection('losses', s_cost)
                self.loss += cost
                #deep_supervision
        self.fuse_output = tf.nn.sigmoid(self.fuse, name='fuse')
        fuse_cost = sigmoid_cross_entropy_balanced(self.fuse, self.edgemaps, name='cross_entropy_fuse')

        self.predictions.append(self.fuse_output)
        f_cost = (self.args.loss_weights * fuse_cost)
        tf.add_to_collection('losses', f_cost)
        self.loss += f_cost

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

