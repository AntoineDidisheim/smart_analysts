import pandas as pd
import numpy as np
from ml_models import LinearAggregator
from ml_models import OneLayerAggregator
from ml_models import MultipleLayerAggregator
from loader import Loader
import time

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


batch_size = 5000
nb_batch_per_epoch = 100
# m1_abs_adam_01
# m1_fullSampleLong_abs_adam_01
# model = LinearAggregator(input_size=13, nb_pred_standing=15, rate_of_saving=1, name='ltest',summary_type="real_data")
# model = OneLayerAggregator(input_size=13, nb_pred_standing=15, rate_of_saving=1, name='m1_fullSample_LayerAllRelu3',summary_type="real_data")
model = MultipleLayerAggregator(start_rate=1,input_size=13, nb_pred_standing=15, rate_of_saving=1,layer_width=[500,250,100,50], name='degrade_5_batch',summary_type="real_data")
model.initialise()
for e in range(100):
    print('------ e', e, '------')
    indices = np.random.permutation(actual_mat.shape[0])
    for i in range(int(np.ceil(len(indices) / batch_size))):
        int_ = indices[(i * batch_size):((i + 1) * batch_size)]
        model.train_sample(train_x=feature_mat[int_, :, :], train_values=values_mat[int_, :], train_actual=actual_mat[int_], epoch=e)
    # model.train_sample(train_x=feature_mat, train_values=values_mat, train_actual=actual_mat, epoch=e)
    outputs, abs_error, sqr_error = model.pred_on_sample_with_summary(test_actual=actual_mat, test_x=feature_mat, test_values=values_mat, epoch=e)
    print('oos m_abs_err', abs_error, '| oos_m_sqr_err', sqr_error)
    model.save_model(epoch=e)
print(model.getParams())
model.sess.close()

# tensorboard --logdir=./logs --host localhost --port 8088
