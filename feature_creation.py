import pandas as pd
import numpy as np
from ml_models import LinearAggregator
from loader import Loader
import time

ibes = Loader.load_ibes_with_feature(reset=False)
feat_name = Loader.feature
ibes  = ibes[ibes['actual']<=ibes['actual'].quantile(0.99)]

ibes.head()
# ibes['consensus_mean']
ibes['pred_number'] = ibes.groupby(['tic', 'tgdate']).cumcount() + 1
ibes['max'] = ibes.groupby(['tic', 'tgdate'])['nb_pred_so_far'].transform('max')

# to select the nb_pred standing
print(ibes['nb_pred_so_far'].describe())
input_size = len(feat_name)
max_pred_standing = 15
min_pred_standing = 5

# adding group_id
ibes['group'] = ibes['tic'] + ibes['tgdate'].astype(str)
t = pd.DataFrame(ibes['group'].unique())
t = t.reset_index().rename(columns={'index': 'group_id', 0: 'group'})
ibes = ibes.merge(t, how='left')


all_feats = np.zeros(shape=(0,max_pred_standing,input_size))
all_sol= np.zeros(shape=(0,6))
all_values = np.zeros(shape=(0,max_pred_standing))
all_actual = np.zeros(shape=(0))

for k in range(5,200):
    print('-----------------', k, '------------------')
    start_time = time.time()
    temp = ibes[(ibes['pred_number'] > k-max_pred_standing) & (ibes['pred_number'] <= k) & (ibes['max'] >= k)].copy()
    temp = temp[feat_name + ['value', 'actual','group_id']+['tic','andate','tgdate','consensus_mean','consensus_median','id']]
    temp.head(25)

    g = temp.groupby('group_id').cumcount()
    mat = (temp.set_index(['group_id', g])
           .unstack(fill_value=0)
           .stack().groupby(level=0)
           .apply(lambda x: x.values.tolist())
           .tolist())
    mat = np.array(mat)
    mat.shape

    # first we remove the last 6 ones that are for comparing consenus and solutions
    mat_sol = mat[:,:,(len(feat_name)+2):]
    # we take the last one only as it is the most recent (i.e., the one used for the prediciton)
    mat_sol = mat_sol[:,-1,:]
    mat_sol.shape
    # not forgeting to remove it from the other one of course
    mat = mat[:,:,:-6]
    # now we can plit the features, values and actuals
    # spliting by removing the last value
    mat_feat = mat[:, :, :-2]
    mat_values = mat[:, :, -2:]
    mat_actual = mat_values[:,:,-1]
    mat_actual = mat_actual[:,0]
    mat_values = mat_values[:,:,-2]

    comp_mat = np.zeros(shape=(mat_feat.shape[0], max_pred_standing - mat_feat.shape[1], mat_feat.shape[2]))
    mat_feat = np.concatenate((mat_feat, comp_mat), axis=1)

    comp_mat = np.zeros(shape=(mat_values.shape[0], max_pred_standing - mat_values.shape[1]))
    mat_values = np.concatenate((mat_values, comp_mat), axis=1)

    all_feats = np.concatenate((all_feats, mat_feat), axis=0)
    all_values = np.concatenate((all_values, mat_values), axis=0)
    all_actual = np.concatenate((all_actual,mat_actual))
    all_sol = np.concatenate((all_sol,mat_sol),axis=0)
    print('done in: ', (time.time()-start_time)/60)

np.save(file='data/int_out/features_mat',arr=all_feats)
np.save(file='data/int_out/values_mat',arr=all_values)
np.save(file='data/int_out/actual_mat',arr=all_actual)
np.save(file='data/int_out/sol_mat',arr=all_sol)

########### now we have finish the traning set, we can go to the testing version (real-life-ish one)
### the first one could use one particluar analyt pred multiplte time if it has multiple combinasion of previous pred that are valide
### this version will make one per pred by taking only the biggest one

# # start by taking the ones with at least 5 pred standings (min for algorithm)
# temp = ibes[(ibes['pred_number'] > min_pred_standing)].copy()

ibes.shape

ibes[ibes.pred_number>5].shape


