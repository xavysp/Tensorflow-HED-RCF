

# import os, sys
import tensorflow as tf
import matplotlib.pyplot as plt
import functools
import time

from models.HED import hed_net
from models.RCF import rcf_net

from utilities.utls import *
from utilities.dataset_manager import (data_parser,
                                  get_training_batch,get_validation_batch, visualize_result)

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

    def run(self, sess):
        with tf.Graph().as_default():
            global_step = tf.train.get_or_create_global_step()

        if not self.init:
            return
        if self.args.dataset4training=="BSDS" and not self.args.use_trained_weights:
            print("While you are using {} dataset  it is necessary use a trained VGG mondel".format(
                self.args.dataset_name))
            # sys.exit()
        train_data = data_parser(self.args)  # train_data = "files_path": train_samples,"n_files": n_train,
                # "train_indices": train_ids,"validation_indices": valid_ids

        self.model.train_model(sess)

        if self.args.lr_scheduler is None:
            global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
            # learning_rate=tf.convert_to_tensor(self.args.learning_rate,dtype=tf.float32)
            self.lr = tf.placeholder(tf.float32, None, name="learning_rate")
        else:
            self.lr = self.args.learning_rate

        if self.args.optimizer.lower()=="adamw":
            # opt = tf.train.AdamOptimizer(learning_rate)
            opt = tf.contrib.opt.AdamWOptimizer(weight_decay = self.args.weight_decay, learning_rate=self.lr)
        elif self.args.optimizer.lower()=="momentum":
            opt =tf.train.MomentumOptimizer(learning_rate=self.lr,
                                                  momentum=0.9)
        elif self.args.optimizer.lower()=="adam":
            # just for adam optimazer
            opt = tf.train.AdamOptimizer(self.lr)
        elif self.args.optimizer.upper()=="GDO" and self.args.model_name.upper()=='HED':
            with tf.control_dependencies([self.model.mean_average_op]):
                opt = tf.train.GradientDescentOptimizer(self.lr)
                grads = opt.compute_gradients(self.model.all_losses)
            apply_grad_op = opt.apply_gradients(grads,global_step=global_step)
            var_avg = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
            with tf.control_dependencies([apply_grad_op]):
                trainG = var_avg.apply(tf.trainable_variables())
        elif self.args.optimizer.upper() == "GDO" and self.args.model_name.upper() == 'RCF':
            pass
        else:
            raise NotImplementedError

        if not self.args.optimizer.upper()=="GDO" and self.args.model_name.upper()=='HED':
            trainG = opt.minimize(self.model.all_losses, global_step=global_step) if self.args.lr_scheduler is not None \
            else opt.minimize(self.model.all_losses)# like hed
        elif not self.args.optimizer.upper()=="GDO" and self.args.model_name.upper()=='RCF':
            pass
        else:
            raise NotImplementedError

        saver = tf.train.Saver(max_to_keep=7)
        sess.run(tf.global_variables_initializer())
        # here to recovery previous training
        if self.args.use_previous_trained:
            model_path = os.path.join(self.args.checkpoint_dir,
                                      self.args.model_name.lower() + '_' + self.args.dataset4training.lower())
            model_path = os.path.join(model_path, 'train')
            if not os.path.exists(model_path) or len(os.listdir(model_path))==0: # :
                ini = 1
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
            ini = 1
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
        prev_lr = self.args.learning_rate
        for idx in range(ini, maxi):

            x_batch, y_batch,_ = get_training_batch(self.args, train_data)


            run_metadata = tf.RunMetadata()
            if idx%5000==0 and idx<5000*2+1:
                prev_lr = prev_lr/10
                print('current learning rate using in training: ',prev_lr)

            _,summary, loss,pred_maps, all_loss= sess.run([trainG,
                                        self.model.merged_summary,
                                         self.model.loss, self.model.predictions,self.model.all_losses],
                                        feed_dict={self.model.input: np.array(x_batch),
                                                   self.model.edgemaps: np.array(y_batch),self.lr:prev_lr})

            self.model.train_writer.add_run_metadata(run_metadata,
                                                     'step{:06}'.format(idx))
            self.model.train_writer.add_summary(summary, idx)

            print(time.ctime(),'[{}/{}]'.format(idx, maxi),' TRAINING loss: %.5f'%loss,' AllLoss: %.5f'%all_loss,
                  'lr: ',prev_lr)

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
                res_list=[]
                res_list.append(x_batch[2][:,:,0:3])
                for i in range(len(pred_maps)):
                    tmp = pred_maps[i][2]
                    res_list.append(tmp)
                res_list.append(y_batch[2])
                vis_imgs = visualize_result(res_list,self.args)

                fig.suptitle("Epoch:" + str(idx + 1) + " Loss:" + '%.5f' % loss + " training")
                # plt.imshow(np.uint8(img))
                fig.add_subplot(1,1,1)
                plt.imshow(np.uint8(vis_imgs))

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
