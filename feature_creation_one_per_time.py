import pandas as pd
import numpy as np
from ml_models import LinearAggregator
from loader import Loader
import time

perc_to_remove = 5

ibes = Loader.load_ibes_with_feature(reset=False)
feat_all = Loader.feature
feat_std = Loader.feature_to_std
feat_simple = []
for f in feat_all:
    if f not in feat_std:
        feat_simple.append(f)

ibes = ibes[ibes['actual'] <= ibes['actual'].quantile(0.99)]

ibes.head()
# ibes['consensus_mean']
ibes['pred_number'] = ibes.groupby(['tic', 'tgdate']).cumcount() + 1
ibes['max'] = ibes.groupby(['tic', 'tgdate'])['nb_pred_so_far'].transform('max')

# removing the train sample
start_year = ibes['andate'].max() - pd.DateOffset(years=1)
ibes = ibes[ibes['andate'] >= start_year]

# saving ibes_original
ibes_original = ibes.copy()


for nb_sim in range(10):
    # re-initialised the ibes file
    ibes =ibes_original.copy()

    # remove random percentage of analyst
    unique_analyst = ibes['analyst'].unique()
    nb_to_remove = int(np.ceil(len(unique_analyst)*perc_to_remove/100))
    analyst_to_remove = np.random.choice(a=unique_analyst,size=nb_to_remove)

    # to select the nb_pred standing
    print(ibes['nb_pred_so_far'].describe())
    input_size = len(feat_all)
    input_std_size = len(feat_std)
    input_simple_size = len(feat_simple)
    max_pred_standing = 15
    min_pred_standing = 5

    # adding group_id
    ibes['group'] = ibes['tic'] + ibes['tgdate'].astype(str)
    t = pd.DataFrame(ibes['group'].unique())
    t = t.reset_index().rename(columns={'index': 'group_id', 0: 'group'})
    ibes = ibes.merge(t, how='left')

    all_feats = np.zeros(shape=(0, max_pred_standing, input_size)).astype(np.float32)
    all_feats_std = np.zeros(shape=(0, max_pred_standing, input_std_size)).astype(np.float32)
    all_feats_std_original = np.zeros(shape=(0, max_pred_standing, input_std_size)).astype(np.float32)
    all_feats_simple = np.zeros(shape=(0, max_pred_standing, input_simple_size)).astype(np.float32)

    all_sol = np.zeros(shape=(0, 6))
    all_values = np.zeros(shape=(0, max_pred_standing)).astype(np.float32)
    all_actual = np.zeros(shape=(0)).astype(np.float32)

    for k in range(5, 67):
        print('-----------------', k, '------------------')
        start_time = time.time()


        def create_the_mat(k, feat_name):
            temp = ibes[(ibes['pred_number'] > k - max_pred_standing) & (ibes['pred_number'] <= k) & (ibes['max'] >= k)].copy()
            temp = temp[feat_name + ['value', 'actual', 'group_id'] + ['tic', 'andate', 'tgdate', 'consensus_mean', 'consensus_median', 'id']]

            g = temp.groupby('group_id').cumcount()
            mat = (temp.set_index(['group_id', g])
                   .unstack(fill_value=0)
                   .stack().groupby(level=0)
                   .apply(lambda x: x.values.tolist())
                   .tolist())
            mat = np.array(mat)
            mat.shape

            # first we remove the last 6 ones that are for comparing consenus and solutions
            mat_sol_ = mat[:, :, (len(feat_name) + 2):]
            # we take the last one only as it is the most recent (i.e., the one used for the prediciton)
            mat_sol_ = mat_sol_[:, -1, :]
            mat_sol_.shape
            # not forgeting to remove it from the other one of course
            mat = mat[:, :, :-6]
            # now we can plit the features, values and actuals
            # spliting by removing the last value
            mat_feat_ = mat[:, :, :-2]
            mat_values_ = mat[:, :, -2:]
            mat_actual_ = mat_values_[:, :, -1]
            mat_actual_ = mat_actual_[:, 0]
            mat_values_ = mat_values_[:, :, -2]

            comp_mat = np.zeros(shape=(mat_feat_.shape[0], max_pred_standing - mat_feat_.shape[1], mat_feat_.shape[2]))
            mat_feat_ = np.concatenate((mat_feat_, comp_mat), axis=1)

            comp_mat = np.zeros(shape=(mat_values_.shape[0], max_pred_standing - mat_values_.shape[1]))
            mat_values_ = np.concatenate((mat_values_, comp_mat), axis=1)

            mat_feat_ = np.nan_to_num(mat_feat_.astype(np.float32))
            mat_values_ = np.nan_to_num(mat_values_.astype(np.float32))
            mat_actual_ = np.nan_to_num(mat_actual_.astype(np.float32))
            # mat_sol_ = np.nan_to_num(mat_sol_).astype(np.float32)

            return mat_values_, mat_actual_, mat_feat_, mat_sol_


        # first we do the simple feat, here we also do all the non-feat related one
        mat_values, mat_actual, mat_feat, mat_sol = create_the_mat(k=k, feat_name=feat_simple)
        all_feats_simple = np.concatenate((all_feats_simple, mat_feat), axis=0)
        all_values = np.concatenate((all_values, mat_values), axis=0)
        all_actual = np.concatenate((all_actual, mat_actual))
        all_sol = np.concatenate((all_sol, mat_sol), axis=0)

        # now the std that are not std
        mat_values, mat_actual, mat_feat, mat_sol = create_the_mat(k=k, feat_name=feat_std)
        all_feats_std_original = np.concatenate((all_feats_std_original, mat_feat), axis=0)

        # finally we std the feats
        # first we compute the mean and var
        m = np.mean(mat_feat, axis=1)
        s = np.std(mat_feat, axis=1)
        s[s == 0] = 1  # avoid divid by 0

        # then we change the shape for the finalstdardidasation
        m = m.reshape(mat_feat.shape[0], 1, mat_feat.shape[2])
        m = m.repeat(mat_feat.shape[1], axis=1)
        s = s.reshape(mat_feat.shape[0], 1, mat_feat.shape[2])
        s = s.repeat(mat_feat.shape[1], axis=1)
        mat_feat = mat_feat.copy()
        mat_feat = (mat_feat - m) / s  # finalstdardisation

        # now we can add it
        all_feats_std = np.concatenate((all_feats_std, mat_feat), axis=0)

        print('done in: ', (time.time() - start_time) / 60, 'added:', mat_actual.shape[0], 'rows')



    # feature_mat = np.concatenate((all_feats_simple, all_feats_std), axis=2)

    np.save(file='data/int_out/sim/features_mat_simple_sim'+str(nb_sim)+"_perc"+str(perc_to_remove),arr=all_feats_simple)
    np.save(file='data/int_out/sim/features_mat_std_sim'+str(nb_sim)+"_perc"+str(perc_to_remove),arr=all_feats_std)
    np.save(file='data/int_out/sim/features_mat_std_original_sim'+str(nb_sim)+"_perc"+str(perc_to_remove),arr=all_feats_std_original)
    np.save(file='data/int_out/sim/values_mat_sim'+str(nb_sim)+"_perc"+str(perc_to_remove),arr=all_values)
    np.save(file='data/int_out/sim/actual_mat_sim'+str(nb_sim)+"_perc"+str(perc_to_remove),arr=all_actual)
    np.save(file='data/int_out/sim/sol_mat_sim'+str(nb_sim)+"_perc"+str(perc_to_remove),arr=all_sol)
    #
    # ########### now we have finish the traning set, we can go to the testing version (real-life-ish one)
    # # TODO look through the dates on to create or split-time wise
    #
    # df = pd.DataFrame(all_sol)
    # df.iloc[:,1] = pd.to_datetime(df.iloc[:,1],unit='ns')
    # df.iloc[:,2] = pd.to_datetime(df.iloc[:,2])
    # df.columns = ['tic','andate','tgdate','consensus_mean','consensus_median','id']
    # start_test_date_1 = df.andate.max()-pd.DateOffset(years=1)
    # start_test_date_2 = df.andate.max()-pd.DateOffset(years=2)
    #
    # test_1 = np.where(df.andate>=start_test_date_1)[0]
    # train_1 = np.where(df.andate<start_test_date_1)[0]
    #
    # test_2 = np.where(df.andate>=start_test_date_2)[0]
    # train_2 = np.where(df.andate<start_test_date_2)[0]
    #
    # np.save(file='data/int_out/train_1',arr=train_1)
    # np.save(file='data/int_out/train_2',arr=train_2)
    #
    # np.save(file='data/int_out/test_1',arr=test_1)
    # np.save(file='data/int_out/test_2',arr=test_2)
    #
    #
    #
