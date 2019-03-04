import pandas as pd
import numpy as np
from ml_models import MultipleLayerAggregator
from loader import Loader
from train_operation import Trainer
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
import seaborn as sns


ibes = Loader.load_ibes_with_feature()

sns.distplot(ibes['error'],hist=False)
plt.show()
ibes.dtypes
std_analysts = ibes.groupby(['analyst','tic'])['error'].std()
std_analysts.describe()
#
# acutals = ibes.groupby(['analyst','tic'])['actual'].mean()
# std_analysts.describe()


#### CREATING DATA ####
np.random.seed(1234) # to be able to reproduce it
nb_features = 15
nb_analyst =20
nb_sample = 10000
test_sample_size = int(nb_sample*0.1)
features = np.random.normal(size=(nb_sample,nb_analyst,nb_features)).clip(-3,3)
#dim are: sample_sizte, analyst, feature
true_value = 1
actual = np.random.normal(size=nb_sample,loc=true_value)
#now we define the function that creates analyst precision.
pos_row = [0,1,2]
neg_row = [3,4,5]
pos_inter = [0,6]
pos_inter2 = [1,7]
bias_pos =[0,9]
bias_tresh = 0.0
bias_value = true_value*1
pos_beta = 0.1

neg_beta = -0.2
inter_beta=0.15
analyst_std = np.sum(features[:,:,pos_row]*pos_beta,axis=2)+\
              np.sum(features[:,:,neg_row]*neg_beta,axis=2)+\
              np.sum((features[:,:,pos_inter2]*features[:,:,pos_inter]).clip(0,10)*inter_beta,axis=2)
analyst_std = 1+np.max(analyst_std)-analyst_std
analyst_std = analyst_std*0.5/np.mean(analyst_std)


values = true_value+np.random.normal(scale=analyst_std)

for bias_nb in bias_pos:
    temp=(features[:,:,bias_nb]>bias_tresh).reshape(nb_sample,nb_analyst)
    values[temp]=values[temp]+bias_value*1

    consenus_mean = np.mean(values,axis=1)
    consenus_median = np.median(values,axis=1)




model_name = 'l250l100l50_relu_all_rate001_batch100_'
model = MultipleLayerAggregator(start_rate=0.11,input_size=nb_features, nb_pred_standing=nb_analyst, rate_of_saving=1,layer_width=[250,100,50], name=model_name,summary_type='full_sim')
batch_size= 10
model.initialise()


train_size = nb_sample-test_sample_size
nb_batch_per_epoch = int(np.ceil(train_size / batch_size))

# infinite training
ind_train = np.array(range(nb_sample-test_sample_size))
ind_test = np.array(range(nb_sample-test_sample_size,nb_sample))


def print_change_r_2(df):
    df['bench'] = (df['c_mean'] - df['c_mean'].mean()) / df['c_mean'].std()
    df['bench_median'] = (df['c_median'] - df['c_median'].mean()) / df['c_median'].std()
    df['pred_s'] = (df['pred'] - df['pred'].mean()) / df['pred'].std()

    smf.ols(data=df, formula='actual~bench').fit().summary()
    smf.ols(data=df, formula='actual~bench+pred').fit().summary()

    r1 = smf.ols(data=df, formula='actual~bench').fit().rsquared_adj
    r2 = smf.ols(data=df, formula='actual~bench+pred').fit().rsquared_adj

    print("r1",r1,"r2",r2,"r_imp", (r2 - r1) / r2)

e = -1
while e<100:
    e += 1
    print('------ e' + str(e) + '------')
    indices = np.random.permutation(train_size)
    for i in range(nb_batch_per_epoch):
        # selct the random batch forn random indice
        int_ = indices[(i * batch_size):((i + 1) * batch_size)]
        int_ = ind_train[int_]
        # translate this into the indices inside the training set
        model.train_sample(train_x=features[int_, :, :], train_values=values[int_, :], train_actual=actual[int_], epoch=e)
    # model.train_sample(train_x=feature_mat, train_values=values_mat, train_actual=actual_mat, epoch=e)
    outputs, abs_error, sqr_error = model.pred_on_sample_without_summary(test_actual=actual[ind_test], test_x=features[ind_test, :]
                                                                           , test_values=values[ind_test, :], epoch=e)

    df = pd.DataFrame(data={'actual': actual[ind_test], 'c_mean': consenus_mean[ind_test], 'c_median': consenus_median[ind_test], 'pred': outputs})
    df['c_mean_error'] = (df['c_mean'] - df['actual']).abs()
    df['c_median_error'] = (df['c_median'] - df['actual']).abs()
    df['pred_error'] = (df['pred'] - df['actual']).abs()

    print('c_mean_e',df['c_mean_error'].mean().round(3),
          'c_median_error', df['c_median_error'].mean().round(3),
          'pred_error', df['pred_error'].mean().round(3))
    print_change_r_2(df)


    # print('oos m_abs_err ' + str(abs_error) + ' | oos_m_sqr_err ' + str(sqr_error))
    model.save_model(epoch=e)
# double_print(model.getParams())

f = model.pred_simple(test_actual=actual[ind_test],test_x=features[ind_test,:,:],test_values=values[ind_test])
f=outputs


df = pd.DataFrame(data={'actual':actual[ind_test],'c_mean':consenus_mean[ind_test],'c_median':consenus_median[ind_test],'pred':f})
df['c_mean_error'] = (df['c_mean']-df['actual']).abs()
df['c_median_error'] = (df['c_median']-df['actual']).abs()
df['pred_error'] = (df['pred']-df['actual']).abs()
df.describe()



df['bench'] = (df['consensus_mean']-df['actual']).abs()
df['vec'] = (df['pred']-df['actual']).abs()
df[['bench','vec']].describe()





