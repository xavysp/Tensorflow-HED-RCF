"""

"""
__author__ = "Xavier Soria Poma, CVC-UAB"
__email__ = "xsoria@cvc.uab.es / xavysp@gmail.com"
__homepage__="www.cvc.uab.cat/people/xsoria"
__credits__=["HED","RCF","holy-edge"]

import sys
import tensorflow as tf

from utilities.model_parameters import get_parameters
from train import model_trainer
from test import model_tester

def inilialize_tf(gpu_fraction):

    # num_threads = int(os.environ.get('OMP_NUM_THREADS'))
    num_threads = False
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        # return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        return tf.Session(config=tf.ConfigProto())

def main(args=None):

    if args.model_state=='train' or args.model_state=='test':
        sess = inilialize_tf(args.gpu_limit)
        if args.model_state == 'train':
            trainer = model_trainer(args)
            trainer.setup()
            trainer.run(sess)
            sess.close()
        else:
            tester = model_tester(args)
            tester.setup(sess)
            tester.run(sess)
    else:
        print('Teh model_state have to be train or test not: ', args.model_state)
        sys.exit()

if __name__=='__main__':

    arguments = get_parameters()
    arguments = arguments.setting_param()
    main(args=arguments)