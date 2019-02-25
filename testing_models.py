import pandas as pd
import numpy as np
from ml_models import LinearAggregator
from ml_models import MultipleLayerAggregator
from loader import Loader
import time
import tensorflow as tf
from scipy import stats
pd.set_option('display.max_columns', 500)
ibes = Loader.load_ibes_with_feature(reset=False)

ind_test = np.load(file='data/int_out/test_1.npy')
feature_mat_simple = np.load(file='data/int_out/features_mat_simple.npy')
feature_mat_std = np.load(file='data/int_out/features_mat_std.npy')
feature_mat_std_original = np.load(file='data/int_out/features_mat_std_original.npy')
values_mat = np.load(file='data/int_out/values_mat.npy')
actual_mat = np.load(file='data/int_out/actual_mat.npy')
sol_mat = np.load(file='data/int_out/sol_mat.npy')

feature_mat = np.concatenate((feature_mat_simple, feature_mat_std), axis=2)

feature_mat =feature_mat[ind_test,:,:]
actual_mat = actual_mat[ind_test]
values_mat = values_mat[ind_test,:]
sol_mat = sol_mat[ind_test,:]


# model_name = 'l500t4_relu_all_rate1_batch5000'
# model = MultipleLayerAggregator(start_rate=1,input_size=30, nb_pred_standing=15, rate_of_saving=1,layer_width=[500,500,500,500], name=model_name,summary_type="comuter_based")

# model_name = 'linear_try_b10000_'
# model = LinearAggregator(input_size=30, nb_pred_standing=15, rate_of_saving=1, name=model_name,summary_type="comuter_based")
#
# model_name  = '100_sigmoid_all_rate1_batch10000_'
# model = MultipleLayerAggregator(start_rate=1,input_size=30, nb_pred_standing=15, rate_of_saving=1,layer_width=[100], layer_types=[tf.nn.sigmoid], name=model_name,summary_type="real_data")

model_name  = 'l500l250l100l50_relu_all_rate1_batch10000_'
model = MultipleLayerAggregator(start_rate=1,input_size=30, nb_pred_standing=15, rate_of_saving=1,layer_width=[500,250,100,50], layer_types=[tf.nn.relu], name=model_name,summary_type="real_data")

model.load()
# w=model.W.eval(model.sess)
# w=np.abs(w)
# w=w/np.sum(w)
# feat_name = Loader.feature
# feat_name = feat_name+[x + "_standardized" for x in feat_name]
# f_imp =pd.DataFrame(index=np.array(feat_name),data=w.reshape(-1,1))
# f_imp.sort_values(0)


# pred = model.pred_simple(test_actual=actual_mat, test_x=feature_mat, test_values=values_mat)

f = model.pred_simple(test_actual=actual_mat,
                          test_x=feature_mat,
                          test_values=values_mat)



df = pd.DataFrame(sol_mat,columns=['tic','andate','tgdate','consensus_mean','consensus_median','id'])

df['pred'] = f
df['actual']=actual_mat
df['andate'] = pd.to_datetime(df['andate'], format='%Y%m%d')
df['tgdate'] = pd.to_datetime(df['tgdate'], format='%Y%m%d')

df['pred_error'] = (df['pred']-df['actual']).abs()
df['min_error'] = df.groupby(['id'])['pred_error'].transform('min')
df.iloc[:,3:] = df.iloc[:,3:].astype(np.float32)
df.describe()
df.head()

df.head()

df=df.groupby(['andate','tic'])['consensus_mean','consensus_median','actual','pred'].min().reset_index(drop=True)


# saving the perf_df
df.to_csv('data/perf_df/'+model_name+'.csv')


# df =  df[df['actual']<=df['actual'].quantile(0.99)]

df = df[~pd.isnull(df['actual'])]
err_mean_sqr = (np.sum((df['consensus_mean']-df['actual'])**2))/(sum(~pd.isnull(df['actual'])))
err_median_sqr = (np.sum((df['consensus_median']-df['actual'])**2))/(sum(~pd.isnull(df['actual'])))
err_model_sqr = (np.sum((df['pred']-df['actual'])**2))/(sum(~pd.isnull(df['actual'])))
print('----','sqr error','----')
print('mean consensus  ', err_mean_sqr)
print('median consensus', err_median_sqr)
print('model           ', err_model_sqr)
print('improvement     ', np.round(  (1-err_model_sqr/min(err_median_sqr,err_mean_sqr))*100,2), '%')


err_mean_abs = np.sum(np.abs((df['consensus_mean']-df['actual'])))/(sum(~pd.isnull(df['actual'])))
err_median_abs = np.sum(np.abs((df['consensus_median']-df['actual'])))/(sum(~pd.isnull(df['actual'])))
err_model_abs = np.sum(np.abs((df['pred']-df['actual'])))/(sum(~pd.isnull(df['actual'])))
print('----','abs error','----')
print('mean consensus  ', err_mean_abs)
print('median consensus', err_median_abs)
print('model           ', err_model_abs)
print('improvement     ', np.round(  (1-err_model_abs/min(err_median_abs,err_mean_abs))*100,2), '%')

