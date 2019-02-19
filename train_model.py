import numpy as np
from ml_models import MultipleLayerAggregator
from os.path import expanduser
import logging
import os
import random

feature_mat = np.load(file='data/int_out/features_mat.npy')
values_mat = np.load(file='data/int_out/values_mat.npy')
actual_mat = np.load(file='data/int_out/actual_mat.npy')

feature_mat = np.nan_to_num(feature_mat)
values_mat = np.nan_to_num(values_mat)
actual_mat = np.nan_to_num(actual_mat)

actual_mat = actual_mat.astype(np.float32)
feature_mat = feature_mat.astype(np.float32)
values_mat = values_mat.astype(np.float32)


feature_mat = np.nan_to_num(feature_mat)
values_mat = np.nan_to_num(values_mat)
actual_mat = np.nan_to_num(actual_mat)

# adding the standardized input
m = np.mean(feature_mat, axis=2)
s = np.std(feature_mat, axis=2)
s[s == 0] = 1  # avoid divid by 0
m = m.reshape(feature_mat.shape[0], feature_mat.shape[1], 1)
m = m.repeat(feature_mat.shape[2], axis=2)
s = s.reshape(feature_mat.shape[0], feature_mat.shape[1], 1)
s = s.repeat(feature_mat.shape[2], axis=2)

reg_input = (feature_mat - m) / s
reg_input.shape
feature_mat.shape
feature_mat = np.concatenate((feature_mat, reg_input), axis=2)

feature_mat.shape

batch_size = 1000
nb_batch_per_epoch = 1000
# m1_abs_adam_01
# m1_fullSampleLong_abs_adam_01
# model = LinearAggregator(start_rate=1, input_size=40, nb_pred_standing=15, rate_of_saving=1, name='linear_std_data',summary_type="real_data")

home = expanduser("~")

model_name = 'l500t4_relu_all_rate1_batch1000'

logging.basicConfig(filename=os.path.join(home, model_name + '.log'), level=logging.DEBUG)



model = MultipleLayerAggregator(start_rate=1,input_size=50, nb_pred_standing=15, rate_of_saving=1,layer_width=[500,500,500,500], name=model_name,summary_type="comuter_based")


def double_print(str):
    """
    new print function that does console print and ssh print
    :param str: string to print
    """
    print(str)
    logging.debug(str)

#fixing the seed for the sample test
random.seed(1234)
indices = np.random.permutation(actual_mat.shape[0])
test_sample_int_ = indices[0:100000]
model.initialise()
for e in range(10000):
    double_print('------ e'+ str(e)+ '------')
    indices = np.random.permutation(actual_mat.shape[0])
    # for i in range(int(np.ceil(len(indices) / batch_size))):
    for i in range(nb_batch_per_epoch):
        int_ = indices[(i * batch_size):((i + 1) * batch_size)]
        model.train_sample(train_x=feature_mat[int_, :, :], train_values=values_mat[int_, :], train_actual=actual_mat[int_], epoch=e)
    # model.train_sample(train_x=feature_mat, train_values=values_mat, train_actual=actual_mat, epoch=e)
    outputs, abs_error, sqr_error = model.pred_on_sample_with_summary(test_actual=actual_mat[test_sample_int_], test_x=feature_mat[test_sample_int_, :]
                                                                      , test_values=values_mat[test_sample_int_,:], epoch=e)
    double_print('oos m_abs_err '+ str(abs_error)+ ' | oos_m_sqr_err '+ str(sqr_error))
    model.save_model(epoch=e)
# double_print(model.getParams())
model.sess.close()

# tensorboard --logdir=./logs --host localhost --port 8088
