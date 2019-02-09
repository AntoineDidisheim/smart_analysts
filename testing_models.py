import pandas as pd
import numpy as np
from ml_models import LinearAggregator
from loader import Loader
import time
from scipy import stats

ibes = Loader.load_ibes_with_feature(reset=False)


feature_mat = np.load(file='data/int_out/features_mat.npy')
values_mat = np.load(file='data/int_out/values_mat.npy')
actual_mat = np.load(file='data/int_out/actual_mat.npy')
sol_mat = np.load(file='data/int_out/sol_mat.npy')

model = LinearAggregator(input_size=13, nb_pred_standing=15, rate_of_saving=1,name='m1_fullSampleLong_abs_adam_01')

model.load()
w=model.W.eval(model.sess)
w=np.abs(w)
w=w/np.sum(w)
feat_name = Loader.feature_to_use
f_imp =pd.DataFrame(index=np.array(feat_name),data=w.reshape(-1,1))
f_imp.sort_values(0)

# converting it to float and removing nan
values_mat =values_mat.astype(np.float32)
feature_mat = feature_mat.astype(np.float32)
actual_mat = actual_mat.astype(np.float32)
values_mat = np.nan_to_num(values_mat)
feature_mat = np.nan_to_num(feature_mat)
actual_mat = np.nan_to_num(actual_mat)

pred = model.pred_simple(test_actual=actual_mat, test_x=feature_mat, test_values=values_mat)



df = pd.DataFrame(sol_mat,columns=['tic','andate','tgdate','consensus_mean','consensus_median','id'])
df['pred'] = pred
df['actual']=actual_mat
df['andate'] = pd.to_datetime(df['andate'], format='%Y%m%d')
df['tgdate'] = pd.to_datetime(df['tgdate'], format='%Y%m%d')

df['pred_error'] = (df['pred']-df['actual']).abs()
df['min_error'] = df.groupby(['id'])['pred_error'].transform('min')


err_mean_sqr = np.sqrt(np.sum((df['consensus_mean']-df['actual'])**2))/(sum(~pd.isnull(df['actual'])))
err_median_sqr = np.sqrt(np.sum((df['consensus_median']-df['actual'])**2))/(sum(~pd.isnull(df['actual'])))
err_model_sqr = np.sqrt(np.sum((df['pred']-df['actual'])**2))/(sum(~pd.isnull(df['actual'])))
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

# def compute_perf(true_v, estimated_v):
#     pre = np.sqrt(sum((true_v - estimated_v)**2)) / len(true_v)
#     return pre
#
#
# def compute_perc(perc, score):
#     pre = stats.percentileofscore(a=perc, score=np.mean(score))
#     return pre
#
#
# def compute_performance(name_of_tg,test_s_,weight_by_id = ['tic', 'analyst'], weight_by_id_2 = ['tic'], verbose=False):
#     # example of a pre to analyse with the median consensu
#     pre_to_evaluate = test_s_.groupby('tic').apply(
#         lambda x: compute_perf(true_v=x['actual'], estimated_v=x[name_of_tg]))
#     pre_to_evaluate = pre_to_evaluate.reset_index().rename(columns={0: 'pre_to_evaluate'})
#
#     # first we construct the weighting scheme
#
#     pre_analysts_firm = test_s_.groupby(weight_by_id).apply(
#         lambda x: compute_perf(true_v=x['actual'], estimated_v=x['value']))
#     pre_analysts_firm = pre_analysts_firm.reset_index().rename(columns={0: 'analyst_pre'})
#
#     # merge the analyst and pre measure
#     pre_analysts_firm = pre_analysts_firm.merge(pre_to_evaluate, how='left')
#     pre_analysts_firm.head()
#
#     # computing the percentile!
#     final_score = pre_analysts_firm.groupby('tic').apply(lambda x: compute_perc(x['analyst_pre'], x['pre_to_evaluate']))
#
#     # computing the weights
#     weights_df = test_s_.groupby(weight_by_id_2)['value'].count().reset_index()
#
#
#     # expanding the final score for final merge
#     final_score = final_score.reset_index().rename(columns={0: 'score_pos'})
#
#     # final merge
#     final_score = final_score.merge(weights_df, how='left')
#     qts = final_score['score_pos'].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).values
#
#     res = {'unweighted_mean': final_score['score_pos'].mean(),
#            'weighted_mean': np.sum(final_score['score_pos'] * final_score['value']) / np.sum(final_score['value']),
#            'std': final_score['score_pos'].mean(), 'quantiler': qts}
#     if verbose:
#         print('performance '+name_of_tg)
#         print(res)
#     return res
#
# ibes = ibes[['analyst','tic','tgdate','andate','value']]
#
# df =df.merge(ibes,how='left',on=['tic','tgdate','andate'])
#
# df.shape
# ibes.shape
# df['value']
# model_perf = compute_performance(name_of_tg='pred',test_s_=df,verbose=True)
# consensus_median_perf = compute_performance(name_of_tg='consensus_median',test_s_=df,verbose=True)
# consensus_mean_perf = compute_performance(name_of_tg='consensus_median',test_s_=df,verbose=True)