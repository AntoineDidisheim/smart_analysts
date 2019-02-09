from loader import Loader
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from scipy import stats
from sklearn import ensemble, model_selection

pd.set_option('display.max_columns', 500)

df_pred = Loader.load_ibes(reset=False)
comp_id = Loader.comp_id

# creating the pairs for the cross validation (adding the id per)
pairs = df_pred[['tic', 'tgdate']].drop_duplicates().reset_index(drop=True)
pairs['u'] = pairs.index
df_pred = df_pred.merge(pairs)

# consensus data
df_pred = df_pred.sort_values(['tic','andate'])
df_pred['consensus_forecast'] = df_pred.groupby(['tic', 'tgdate'])['value'].apply(lambda x: x.expanding().mean())
df_pred['consensus_forecast_L1'] = df_pred.groupby(['tic','tgdate'])['consensus_forecast'].shift(1)
df_pred['consensus_median'] = df_pred.groupby(['tic', 'tgdate'])['value'].apply(lambda x: x.expanding().median())
df_pred['consensus_median_L1'] = df_pred.groupby(['tic','tgdate'])['consensus_median'].shift(1)
df_pred['consensus_std'] = df_pred.groupby(['tic', 'tgdate'])['value'].apply(lambda x: x.expanding().std())
df_pred['consensus_std_L1'] = df_pred.groupby(['tic','tgdate'])['consensus_std'].shift(1)

df_pred['nb_forecast'] = df_pred.groupby(['tic', 'tgdate'])['value'].transform('count')
df_pred = df_pred[df_pred['nb_forecast']>=20]

# df_pred = df_pred.merge(right=df, on=['tic', 'date'], how='inner')
sic_id = Loader.sic_id
other_id = ['u', 'analyst', 'value', 'actual', 'tic']
l_id = ['actual_L1', 'actual_L2', 'actual_L3', 'actual_L4','quarterly_ret']
# consensus_id = ['consensus_forecast', 'consensus_median', 'consensus_std']
consensus_id = ['consensus_forecast_L1', 'consensus_median_L1', 'consensus_std_L1']
news_id = ['news_sum', 'news_sum_r', 'news_sum_r30', 'news_sum_r250']
all_id = other_id + l_id + consensus_id + comp_id+sic_id

pred_id = l_id + consensus_id+comp_id

'consensus_forecast+consensus_median+consensus_std'
news_c = 'news_sum+news_sum_r+news_sum_r30+news_sum_r250+news_sum+news_sum_quarterly'
comp_c = 'prccq+prchq+prclq+oepsxy+oepf12+epspiq+niq+dlttq'

ols(data=df_pred, formula='actual~actual_L1+actual_L2+actual_L3+actual_L4+quarterly_ret+consensus_forecast+consensus_median+consensus_std').fit().summary()
ols(data=df_pred, formula='actual~news_sum_quarterly').fit().summary()

# standardisation of comp_id
df_pred[comp_id]=df_pred[comp_id].div(df_pred['price'], axis=0)




# spliting the sample by pair year/data
test_u = pairs.sample(frac=0.2, replace=False).u
train_u = pairs.u[~pairs.u.isin(test_u)]

all_s = df_pred[all_id].dropna()
# removing quantiles
# q=all_s.actual.quantile([0.05,0.95]).values
# all_s = all_s[(all_s.actual>=q[0]) & (all_s.actual<=q[1])]
train_s = all_s[all_s.u.isin(train_u)].drop(columns='u')
test_s = all_s[~all_s.u.isin(train_u)].drop(columns='u')


def compute_perf(true_v, estimated_v):
    pre = sum(np.abs(true_v - estimated_v)) / len(true_v)
    return pre


def compute_perc(perc, score):
    pre = stats.percentileofscore(a=perc, score=np.mean(score))
    return pre



temp = train_s[pred_id + ['actual']].drop_duplicates()
model = ensemble.RandomForestRegressor(n_estimators=100, max_depth=50, n_jobs=-1)

# fit hyper
param_dist = {"max_depth": [25, 50, 100, None],
              "n_estimators": [100, 500, 1000],
              "criterion": ["mse"]}

from sklearn.model_selection import GridSearchCV

# random_search = GridSearchCV(model, param_grid =param_dist, cv=2)
# print('start')
# random_search.fit(X=temp[pred_id],y=temp['actual'])
# random_search.cv_results_
# random_search.best_score_
# random_search.best_params_
# end hyper


model.fit(X=temp[pred_id], y=temp['actual'])

feat_imp = pd.DataFrame(data=model.feature_importances_, index=pred_id)
print(feat_imp.sort_values(0,ascending=False))
# rw_pre = (sum(np.abs(X_test['actual_L1']-y_test)))/len(y_test)
# pre_algo = (sum(np.abs(model.predict(X=X_test)-y_test)))/len(y_test)
# pre = all_test.groupby('analyst').apply(lambda x: (sum(np.abs(x['actual']-x['value'])))/len(x['value']))


test_s['model'] = model.predict(X=test_s[pred_id])
# par = 0.2
# test_s['model'][(test_s['model']>test_s['consensus_forecast']*(1+par))] = test_s['consensus_forecast'][(test_s['model']>test_s['consensus_forecast']*(1+par))]
# test_s['model'][(test_s['model']<test_s['consensus_forecast']*(0+par))] = test_s['consensus_forecast'][(test_s['model']<test_s['consensus_forecast']*par)]


# # clipping
# t = train_s.groupby('tic')['actual'].apply(lambda x: x.min()*1).reset_index().rename(columns={'actual':'min_a'})
# test_s = test_s.merge(t,how='left')
# t = train_s.groupby('tic')['actual'].apply(lambda x: x.max()*1.2).reset_index().rename(columns={'actual':'max_a'})
# test_s = test_s.merge(t,how='left')
# test_s['model'][test_s['model']<test_s['min_a']] = test_s['min_a'][test_s['model']<test_s['min_a']]
# test_s['model'][test_s['model']>test_s['max_a']] = test_s['max_a'][test_s['model']>test_s['max_a']]
# test_s['actual'][test_s['model']>test_s['max_a']]
# test_s[['actual','model','max_a']][test_s['model']>test_s['max_a']].head(50)

def compute_performance(name_of_tg,test_s_,weight_by_id = ['tic', 'analyst'], weight_by_id_2 = ['tic'], verbose=False):
    # example of a pre to analyse with the median consensu
    pre_to_evaluate = test_s_.groupby('tic').apply(
        lambda x: compute_perf(true_v=x['actual'], estimated_v=x[name_of_tg]))
    pre_to_evaluate = pre_to_evaluate.reset_index().rename(columns={0: 'pre_to_evaluate'})

    # first we construct the weighting scheme

    pre_analysts_firm = test_s_.groupby(weight_by_id).apply(
        lambda x: compute_perf(true_v=x['actual'], estimated_v=x['value']))
    pre_analysts_firm = pre_analysts_firm.reset_index().rename(columns={0: 'analyst_pre'})

    # merge the analyst and pre measure
    pre_analysts_firm = pre_analysts_firm.merge(pre_to_evaluate, how='left')
    pre_analysts_firm.head()

    # computing the percentile!
    final_score = pre_analysts_firm.groupby('tic').apply(lambda x: compute_perc(x['analyst_pre'], x['pre_to_evaluate']))

    # computing the weights
    weights_df = test_s_.groupby(weight_by_id_2)['value'].count().reset_index()


    # expanding the final score for final merge
    final_score = final_score.reset_index().rename(columns={0: 'score_pos'})

    # final merge
    final_score = final_score.merge(weights_df, how='left')
    qts = final_score['score_pos'].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).values

    res = {'unweighted_mean': final_score['score_pos'].mean(),
           'weighted_mean': np.sum(final_score['score_pos'] * final_score['value']) / np.sum(final_score['value']),
           'std': final_score['score_pos'].mean(), 'quantiler': qts}
    if verbose:
        print('performance '+name_of_tg)
        print(res)
    return res

compute_performance(name_of_tg='model',test_s_=test_s,verbose=True)
compute_performance(name_of_tg='consensus_median_L1',test_s_=test_s,verbose=True)
# compute_performance(name_of_tg='consensus_forecast',test_s_=test_s,verbose=True)
compute_performance(name_of_tg='consensus_forecast_L1',test_s_=test_s,verbose=True)