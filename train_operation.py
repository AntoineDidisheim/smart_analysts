import numpy as np
from ml_models import MultipleLayerAggregator
from os.path import expanduser
import logging
import os
import random


class Trainer:
    def __init__(self, model, batch_size, feature_code=1):
        self.model = model
        self.name = self.model.name
        self.feature_code = feature_code
        home = expanduser("~")
        self.batch_size = batch_size
        self.ind_test = 0
        self.ind_train = 0

        logging.basicConfig(filename=os.path.join(home, self.name + '.log'), level=logging.DEBUG)

    def load_data(self):
        feature_mat_simple = np.load(file='data/int_out/features_mat_simple.npy')
        feature_mat_std = np.load(file='data/int_out/features_mat_std.npy')
        feature_mat_std_original = np.load(file='data/int_out/features_mat_std_original.npy')
        values_mat = np.load(file='data/int_out/values_mat.npy')
        actual_mat = np.load(file='data/int_out/actual_mat.npy')
        self.ind_test = np.load(file='data/int_out/test_1.npy')
        self.ind_train = np.load(file='data/int_out/train_1.npy')

        if self.feature_code == 0:
            feature_mat = np.concatenate((feature_mat_simple, feature_mat_std_original), axis=2)
        if self.feature_code == 1:
            feature_mat = np.concatenate((feature_mat_simple, feature_mat_std), axis=2)
        if self.feature_code == 2:
            feature_mat = np.concatenate((feature_mat_simple, feature_mat_std, feature_mat_std_original), axis=2)

        return feature_mat, values_mat, actual_mat

    def double_print(self, str):
        """
        new print function that does console print and ssh print
        :param str: string to print
        """
        print(str)
        logging.debug(str)

    def train(self, initialised=True):
        self.double_print("start_training")
        # fixing the seed for the sample test
        if initialised:
            self.model.initialise()
        else:
            self.model.load()
        # load the data
        feature_mat, values_mat, actual_mat = self.load_data()

        # we define the nb_batch_per_epoch
        train_size = len(self.ind_train)
        nb_batch_per_epoch = int(np.ceil(train_size / self.batch_size))

        # infinite training
        e = -1
        while True:
            e += 1
            self.double_print('------ e' + str(e) + '------')
            indices = np.random.permutation(train_size)
            for i in range(nb_batch_per_epoch):
                # selct the random batch forn random indice
                int_ = indices[(i * self.batch_size):((i + 1) * self.batch_size)]
                # translate this into the indices inside the training set
                int_ = self.ind_train[int_]
                self.model.train_sample(train_x=feature_mat[int_, :, :], train_values=values_mat[int_, :], train_actual=actual_mat[int_], epoch=e)
            # model.train_sample(train_x=feature_mat, train_values=values_mat, train_actual=actual_mat, epoch=e)
            outputs, abs_error, sqr_error = self.model.pred_on_sample_with_summary(test_actual=actual_mat[self.ind_test], test_x=feature_mat[self.ind_test, :]
                                                                                   , test_values=values_mat[self.ind_test, :], epoch=e)
            self.double_print('oos m_abs_err ' + str(abs_error) + ' | oos_m_sqr_err ' + str(sqr_error))
            self.model.save_model(epoch=e)
        # double_print(model.getParams())
        self.model.sess.close()

# model = MultipleLayerAggregator(start_rate=1,input_size=50, nb_pred_standing=15, rate_of_saving=1,layer_width=[500,500,500,500], name="m_na,e",summary_type="comuter_based")
# self = trainer(model=model,batch_size=50)
# self.load_data()
# self.ind_test

# tensorboard --logdir=./logs --host localhost --port 8088

