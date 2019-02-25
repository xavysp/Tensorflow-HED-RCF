

# import os, sys
import tensorflow as tf
import matplotlib.pyplot as plt
import functools
import time

from models.HED import hed_net
from models.RCF import rcf_net

from utilities.utls import *
from utilities.dataset_manager import (data_parser,
                                  get_training_batch,get_validation_batch)

MOVING_AVERAGE_DECAY = 0.9999
class model_trainer():

    def __init__(self,args ):
        self.init = True
        self.args = args

    def setup(self):
        try:
            if self.args.model_state=='train':
                is_training=True
            else:
                is_training=False
            if self.args.model_name=='HED':
                self.model = hed_net(self.args)
            elif self.args.model_name=='RCF':
                self.model = rcf_net(self.args)
            else:
                print("Error setting model, {}".format(self.args.model_name))

            print("Done initializing VGG-16")
        except Exception as err:
            print("Error setting up VGG-16, {}".format(err))
            self.init=False

    def expon_decay(self,learning_rate, global_step, decay_steps,
                                 decay_rate, staircase, name):
        if global_step is None:
            raise ValueError("global_step is required for exponential_decay.")
        def decayed_lr(learning_rate,global_step, decay_steps, decay_rate,
                      staircase, name):
            with tf.name_scope(name, "ExponentialDecay",
                               [learning_rate, global_step, decay_steps, decay_rate]) as name:
                learning_rate = tf.convert_to_tensor(learning_rate, name="learning_rate")
                dtype = learning_rate.dtype
                decay_steps = tf.cast(decay_steps, dtype)
                decay_rate = tf.cast(decay_rate, dtype)
                global_step_recomp = tf.cast(global_step, dtype)
                p = global_step_recomp / decay_steps
                if staircase:
                    p = tf.math.floor(p)
                return tf.math.divide(
                        learning_rate, tf.math.pow(decay_rate, p), name=name)

        return functools.partial(decayed_lr, learning_rate, global_step, decay_steps,
                                 decay_rate, staircase, name)

    def lr_scheduler_desc(self,learning_rate,global_step, decay_steps, decay_rate,
                      staircase=False, name=None):
        decayed_lr=self.expon_decay(learning_rate,global_step,decay_steps,\
                              decay_rate,staircase=staircase,name=name)
        if not tf.executing_eagerly():
            decayed_lr =decayed_lr()
        return  decayed_lr


    def run(self, sess):
        with tf.Graph().as_default():
            global_step = tf.train.get_or_create_global_step()

        if not self.init:
            return
        if self.args.dataset4training=="HED-BSDS" and not self.args.use_trained_weights:
            print("While you are using {} dataset  it is necessary use a trained VGG mondel".format(
                self.args.dataset_name))
            # sys.exit()
        train_data = data_parser(self.args)  # train_data = "files_path": train_samples,"n_files": n_train,
                # "train_indices": train_ids,"validation_indices": valid_ids

        self.model.train_model(sess)

        if self.args.lr_scheduler is not None:
            global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        if self.args.lr_scheduler is None:
            # learning_rate=tf.convert_to_tensor(self.args.learning_rate,dtype=tf.float32)
            learning_rate = tf.constant(self.args.learning_rate, dtype=tf.float16)
        elif self.args.lr_scheduler=='asce':
            learning_rate = tf.train.exponential_decay(learning_rate=self.args.learning_rate, \
                                                   global_step=global_step,
                                                   decay_steps=self.args.learning_decay_interval,
                                                   decay_rate=self.args.learning_rate_decay, staircase=True)
        elif self.args.lr_scheduler=='desc':
            learning_rate = self.lr_scheduler_desc(learning_rate=self.args.learning_rate,
                                              decay_rate=self.args.learning_rate_decay,
                                              global_step=global_step,
                            decay_steps=self.args.learning_decay_interval)
        else:
            raise NotImplementedError('Learning rate scheduler type [%s] is not implemented',
                                      self.args.lr_scheduler)

        if self.args.optimizer=="adamw" or self.args.optimizer=="adamW":
            # opt = tf.train.AdamOptimizer(learning_rate)
            opt = tf.contrib.opt.AdamWOptimizer(weight_decay = self.args.weight_decay, learning_rate=learning_rate)
        elif self.args.optimizer=="momentum" or self.args.optimizer=="MOMENTUM":
            opt =tf.train.MomentumOptimizer(learning_rate=self.args.learning_rate,
                                                  momentum=0.9)
        elif self.args.optimizer=="adam" or self.args.optimizer=="ADAM":
            # just for adam optimazer
            opt = tf.train.AdamOptimizer(learning_rate)
        elif self.args.optimizer.upper()=="GDO":
            with tf.control_dependencies([self.model.mean_average_op]):
                opt = tf.train.GradientDescentOptimizer(learning_rate)
                grads = opt.compute_gradients(self.model.all_losses)
            apply_grad_op = opt.apply_gradients(grads,global_step=global_step)
            var_avg = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
            with tf.control_dependencies([apply_grad_op]):
                trainG = var_avg.apply(tf.trainable_variables())

        if not self.args.optimizer.upper()=="GDO":
            trainG = opt.minimize(self.model.all_losses, global_step=global_step) if self.args.lr_scheduler is not None \
            else opt.minimize(self.model.all_losses)# like hed

        saver = tf.train.Saver(max_to_keep=7)
        sess.run(tf.global_variables_initializer())
        # here to recovery previous training
        if self.args.use_previous_trained:
            model_path = os.path.join(self.args.checkpoint_dir,
                                      self.args.model_name.lower() + '_' + self.args.dataset4training.lower())
            model_path = os.path.join(model_path, 'train')
            if not os.path.exists(model_path) or len(os.listdir(model_path))==0: # :
                ini = 0
                maxi = self.args.max_iterations+1
                print('There is not previous trained data for the current model... and')
                print('*** The training process is starting from scratch ***')
            else:
                # restoring using the last checkpoint
                assert (len(os.listdir(model_path)) != 0),'There is not previous trained data for the current model...'
                last_ckpt = tf.train.latest_checkpoint(model_path)
                _,last_inter = os.path.split(last_ckpt)
                last_inter = int(last_inter.replace(self.args.model_name+'-',''))
                saver.restore(sess,last_ckpt)
                ini=last_inter
                maxi=ini+self.args.max_iterations+1 # check
                print('--> Previous model restored successfully: ','training from: ', ini,' to: ',maxi )
        else:
            ini = 0
            maxi = self.args.max_iterations + 1
            print('*** The training process is starting from scratch ***')

        prev_loss=None
        prev_val = None
        # directories for checkpoints
        if self.args.use_nir:
            checkpoint_dir = os.path.join(self.args.checkpoint_dir,
                                          os.path.join(
                                              self.args.model_name.lower() + '_' + self.args.dataset4training.lower() + '_RGBN',
                                              self.args.model_state))
        else:
            checkpoint_dir = os.path.join(self.args.checkpoint_dir,
                                          os.path.join(self.args.model_name.lower() + '_' + self.args.dataset4training.lower(),
                                                       self.args.model_state))
        _ =make_dirs(checkpoint_dir)

        fig = plt.figure()
        for idx in range(ini, maxi):

            x_batch, y_batch,_ = get_training_batch(self.args, train_data)


            run_metadata = tf.RunMetadata()

            _,summary, loss,pred_maps, all_loss= sess.run([trainG,
                                        self.model.merged_summary,
                                         self.model.loss, self.model.predictions,self.model.all_losses],
                                        feed_dict={self.model.input: np.array(x_batch),
                                                   self.model.edgemaps: np.array(y_batch),})

            self.model.train_writer.add_run_metadata(run_metadata,
                                                     'step{:06}'.format(idx))
            self.model.train_writer.add_summary(summary, idx)

            print(time.ctime(),'[{}/{}]'.format(idx, maxi),' TRAINING loss: %.5f'%loss,' AllLoss: %.5f'%all_loss)

            # saving trained parameters
            if prev_loss is not None and (idx>=self.args.save_interval):
                if loss<=(prev_loss-prev_loss*0.05) and loss>=(prev_loss-prev_loss*0.3):
                    saver.save(sess, os.path.join(checkpoint_dir, self.args.model_name), global_step=idx)
                    prev_loss = loss
                    print("parameters saver because of 10% of previous loss ", idx)
                elif idx%self.args.save_interval==0: # 0.48
                    saver.save(sess, os.path.join(checkpoint_dir, self.args.model_name), global_step=idx)
                    prev_loss = loss
                    print("parameters saved when the iteration get 48% of its purpose", idx, "max_iter= ",maxi)

            else:
                if prev_loss==None:
                    saver.save(sess, os.path.join(checkpoint_dir, self.args.model_name), global_step=idx)
                    prev_loss = loss
                    print("parameters saved for the first time ", idx)
                elif(idx % (self.args.save_interval//4) == 0) and \
                            (loss<=(prev_loss-prev_loss*0.05) and loss>=(prev_loss-prev_loss*0.5)):
                    saver.save(sess, os.path.join(checkpoint_dir, self.args.model_name), global_step=idx)
                    prev_loss = loss
                    print("parameters saver because of 10% of previous loss (save_interval//4)", idx)

            # ********* for validation **********
            if (idx+1) % self.args.val_interval== 0:
                pause_show=0.01
                # plt.close()
                # *** recode with restore_rgb fuinction **********
                img = x_batch[2]
                img = img[:,:,0:3]
                img = restore_rgb([self.args.channel_swap,self.args.mean_pixel_values[:3]],img)

                gt_mp= y_batch[2]
                gt_mp = np.squeeze(gt_mp)
                gt_mp = image_normalization(gt_mp)

                p_map1 = pred_maps[0]
                p_map1=p_map1[2,...]
                p_map1 =np.squeeze(p_map1)
                p_map1 = image_normalization(p_map1,0,1)
                p_map1 = image_normalization(p_map1)

                p_map2 = pred_maps[1]
                p_map2 = p_map2[2, ...]
                p_map2 = np.squeeze(p_map2)
                p_map2 = image_normalization(p_map2,0,1)
                p_map2 = image_normalization(p_map2)

                p_map3 = pred_maps[2]
                p_map3 = p_map3[2, ...]
                p_map3 = np.squeeze(p_map3)
                p_map3 = image_normalization(p_map3,0,1)
                p_map3 = image_normalization(p_map3)

                p_map4 = pred_maps[3]
                p_map4 = p_map4[2, ...]
                p_map4 = np.squeeze(p_map4)
                p_map4 = image_normalization(p_map4,0,1)
                p_map4 = image_normalization(p_map4)

                p_map5 = pred_maps[4]
                p_map5 = p_map5[2, ...]
                p_map5 = np.squeeze(p_map5)
                p_map5 = image_normalization(p_map5,0,1)
                p_map5 = image_normalization(p_map5)

                p_map6 = pred_maps[5] # pred_maps[len(pred_maps)-1]
                p_map6 = p_map6[2, ...]
                p_map6 = np.squeeze(p_map6)
                p_map6 = image_normalization(p_map6,0,1)
                p_map6 = image_normalization(p_map6)

                # plt.title("Epoch:" + str(idx + 1) + " Loss:" + '%.5f' % loss + " training")

                fig.suptitle("Epoch:" + str(idx + 1) + " Loss:" + '%.5f' % loss + " training")
                # plt.imshow(np.uint8(img))
                fig.add_subplot(2,4,1)
                plt.imshow(np.uint8(img))
                fig.add_subplot(2, 4, 2)
                plt.imshow(np.uint8(p_map1))
                fig.add_subplot(2, 4, 3)
                plt.imshow(np.uint8(p_map2))
                fig.add_subplot(2, 4, 4)
                plt.imshow(np.uint8(p_map3))
                fig.add_subplot(2, 4, 5)
                plt.imshow(np.uint8(p_map4))
                fig.add_subplot(2, 4, 6)
                plt.imshow(np.uint8(p_map5))
                fig.add_subplot(2, 4, 7)
                plt.imshow(np.uint8(p_map6))
                fig.add_subplot(2, 4, 8)
                plt.imshow(np.uint8(gt_mp))

                print("Evaluation in progress...")
                plt.draw()
                plt.pause(pause_show)

                im, em, _ = get_validation_batch(self.args, train_data)

                summary, error, pred_val = sess.run([self.model.merged_summary, self.model.error,
                                           self.model.fuse],
                                          feed_dict={self.model.input: im, self.model.edgemaps: em})
                if error<=0.13: # all annotation concideration: 0.13 when is greather that 50 <=0.09
                    saver.save(sess, os.path.join(checkpoint_dir, self.args.model_name), global_step=idx)
                    prev_loss = loss
                    print("Parameters saved in the validation stage when its error is <=0.11::", error)
                # save valid result
                # if idx % self.args.save_interval == 0:
                #     imi = restore_rgb([self.args.channel_swap,self.args.mean_pixel_value[:3]],
                #                  im)
                #     save_result(self.args,[imi,em,pred_val])
                self.model.val_writer.add_summary(summary, idx)
                print(time.ctime(),' [{}/{}]'.format(idx, maxi),' VALIDATION error: %.5f'%error,' pError: %.5f'%prev_loss)
                if (idx+1) % (self.args.val_interval*200)== 0:
                    print('updating visualisation')
                    plt.close()
                    fig = plt.figure()

        # plt.show()
        self.model.train_writer.close()
