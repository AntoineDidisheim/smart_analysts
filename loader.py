import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)


class Loader:
    comp_id = ['prccq', 'prchq', 'prclq', 'oepsxy', 'oepf12', 'epspiq', 'niq', 'dlttq']
    sic_id = ['sic1', 'sic2', 'sic3', 'sic4', 'sic5', 'sic6', 'sic7', 'sic8', 'sic9', 'sic10', 'sic11', 'sic12']
    @staticmethod
    def load_ibes(reset=False, with_price_ret=True, with_compustat=True, with_news=True):
        if reset:
            ibes = pd.read_csv('data/ibes_quarterly.csv')
            ibes = ibes.rename(
                columns={'ANALYS': 'analyst', 'ANNDATS': 'andate', 'REVDATS': 'revdate', 'ACTDATS': 'acdats',
                         'OFTIC': 'tic', 'VALUE': 'value',
                         'FPEDATS': 'tgdate',
                         'ACTUAL': 'actual', 'FPI': 'fpi'})

            ibes['andate'] = pd.to_datetime(ibes['andate'], format='%Y%m%d')
            ibes['acdats'] = pd.to_datetime(ibes['acdats'], format='%Y%m%d')
            ibes['tgdate'] = pd.to_datetime(ibes['tgdate'], format='%Y%m%d')

            t = ibes[['tic', 'tgdate', 'actual']].drop_duplicates().sort_values(['tic', 'tgdate'])
            t['actual_L1'] = t.groupby('tic')['actual'].shift(1)
            t['actual_L2'] = t.groupby('tic')['actual'].shift(2)
            t['actual_L3'] = t.groupby('tic')['actual'].shift(3)
            t['actual_L4'] = t.groupby('tic')['actual'].shift(4)
            t = t.drop(columns=['actual'])
            ibes = ibes.merge(t, on=['tic', 'tgdate'], how='left')
            ibes = ibes.sort_values(['tic', 'tgdate'])
            if with_price_ret:
                ibes = Loader.add_quarterly_ret(ibes)
            if with_news:
                ibes = Loader.add_news_data(ibes)
            if with_compustat:
                # ibes = Loader.add_news_data(ibes)
                comp, comp_id = Loader.load_compustat()
                ibes['year'] = ibes['andate'].dt.year
                ibes['quarter'] = 1 + np.floor(ibes['andate'].dt.month / 4)
                ibes = ibes.merge(comp, on=['year', 'quarter', 'tic'], how='left')
                # ibes.to_pickle('data/ibes_processed.p')
            ibes.to_pickle('data/ibes_processed.p')
        else:
            ibes = pd.read_pickle('data/ibes_processed.p')
            ibes.head()

        return ibes

    @staticmethod
    def add_quarterly_ret(ibes):
        ibes.head()
        # creating the short list of dates for the start of quarters
        dt = ibes[['tic', 'tgdate']].drop_duplicates().rename(columns={'tgdate': 'quart_date'})
        dt['date'] = dt['quart_date']
        # loading and processing the price orignal data
        price = pd.read_csv('data/price_all.csv')
        price = price.rename(columns={'TICKER': 'tic', 'PRC': 'price', 'FACPR': 'facpr', 'CFACPR': 'cfacpr'})
        price.date = pd.to_datetime(price.date, format='%Y%m%d')
        # correct neg, as crsp put negative price to indicate which one was used...
        price.price = price.price.apply(np.abs)

        # following crisp procedure to adjust price
        price['price_adj'] = price['price'] / price['cfacpr']

        # by merge I add here the quarter dates
        dt['date'] = dt['quart_date']
        price = price.merge(dt, how='left', on=['tic', 'date'])

        # I create the adjust price at the start of the quarter by nana group and extend
        price['quarter_ref_p'] = np.nan
        price.loc[~pd.isnull(price.quart_date), 'quarter_ref_p'] = price.loc[~pd.isnull(price.quart_date), 'price_adj']
        price['quarter_ref_p'] = price.groupby('tic')['quarter_ref_p'].fillna(method='ffill')

        # finally, the quarterly ret
        price['quarterly_ret'] = ((price['price_adj'] - price['quarter_ref_p']) / price['quarter_ref_p'])

        # now we can merge the usefull columns
        price = price[['price', 'price_adj', 'quarter_ref_p', 'quarterly_ret', 'date', 'tic']]
        ibes = ibes.merge(price, how='left', left_on=['tic', 'andate'], right_on=['tic', 'date'])
        return ibes

    @staticmethod
    def add_news_data(ibes):

        df = Loader.load_data()

        df['news_sum'] = df['sent_p'].replace(0, np.nan) - df['sent_n'].replace(0, np.nan)
        # df['news_sum'] = df['socsent_p'].replace(0, np.nan) - df['socsent_n'].replace(0, np.nan)
        df['abs_ret'] = df.ret.apply(np.abs)
        df['news_dir'] = df['news_sum'].apply(np.sign)
        df['news_sum'] = df['news_sum'].fillna(0)
        df['news_sum_r'] = df.groupby('tic')['news_sum'].rolling(5).mean().reset_index()['news_sum']
        df['news_std_r'] = df.groupby('tic')['news_sum'].rolling(5).std().reset_index()['news_sum']
        df['news_std_r30'] = df.groupby('tic')['news_sum'].rolling(30).std().reset_index()['news_sum']
        df['news_std_r250'] = df.groupby('tic')['news_sum'].rolling(250).std().reset_index()['news_sum']

        df['news_sum_r30'] = df.groupby('tic')['news_sum'].rolling(30).mean().reset_index()['news_sum']
        df['news_sum_r20'] = df.groupby('tic')['news_sum'].rolling(20).mean().reset_index()['news_sum']
        df['news_sum_r250'] = df.groupby('tic')['news_sum'].rolling(250).mean().reset_index()['news_sum']
        df['news_5_minus_250'] = df['news_sum_r'] - df['news_sum_r250']
        df['news_1_minus_250'] = df['news_sum'] - df['news_sum_r250']
        df['news_sum_r250'] = df.groupby('tic')['news_sum'].rolling(250).mean().reset_index()['news_sum']
        df['news_0_minus_30'] = df['news_sum'] - df['news_sum_r30']
        df['news_5_minus_30'] = df['news_sum_r'] - df['news_sum_r30']

        df['soc_sum'] = df['socsent_p'].replace(0, np.nan) - df['socsent_n'].replace(0, np.nan)
        df['soc_dir'] = df['soc_sum'].apply(np.sign)
        df['soc_sum'] = df['soc_sum'].fillna(0)
        df['soc_sum_r'] = df.groupby('tic')['soc_sum'].rolling(5).mean().reset_index()['soc_sum']
        df['soc_sum_r30'] = df.groupby('tic')['soc_sum'].rolling(30).mean().reset_index()['soc_sum']
        df['soc_sum_r250'] = df.groupby('tic')['soc_sum'].rolling(250).mean().reset_index()['soc_sum']
        df['soc_5_minus_250'] = df['soc_sum_r'] - df['soc_sum_r250']
        df['soc_sum_r250'] = df.groupby('tic')['soc_sum'].rolling(250).mean().reset_index()['soc_sum']
        df['soc_0_minus_30'] = df['soc_sum'] - df['soc_sum_r30']

        # adding cumulated return
        df = df.sort_values(['tic', 'date'])
        df['ret_30'] = df.groupby(['tic'])['ret'].apply(lambda x: (x / 100 + 1).rolling(30).agg(lambda y: y.prod()))
        df['ret_5'] = df.groupby(['tic'])['ret'].apply(
            lambda x: (x / 100 + 1).rolling(5).agg(lambda y: y.prod(skipna=True)))

        dt = ibes[['tic', 'tgdate']].drop_duplicates().rename(columns={'tgdate': 'quart_date'})
        dt['date'] = dt['quart_date']

        df = df.merge(dt, how='left', on=['tic', 'date'])
        df['quart_date'] = df['quart_date'].fillna(method='ffill')
        df['news_sum_quarterly'] = df.groupby(['tic', 'quart_date'])['news_sum'].apply(lambda x: x.expanding().mean())

        ibes = ibes.merge(df, how='left', on=['tic', 'date'])
        return ibes

    @staticmethod
    def load_compustat():
        pred_id = ['prccq', 'prchq', 'prclq', 'oepsxy', 'oepf12', 'epspiq', 'niq', 'dlttq']  # actq invchy
        comp = pd.read_csv('data/compustat_quarterly.csv')
        sic_var = [100, 1000, 1500, 1800, 2000, 4000, 5000, 5200, 6000, 7000, 9100, 9900, 10000]
        sn = []
        for i in range(1, len(sic_var)):
            n = 'sic' + str(i)
            comp[n] = 1 * ((comp.sic >= sic_var[i - 1]) & (comp.sic < sic_var[i]))
            sn.append(n)

        comp.head()

        comp.dtypes
        comp.sic = comp.sic.apply(lambda x: str(x))
        t = comp.sic.str.get_dummies()

        for id in pred_id:
            print(id, sum(pd.isnull(comp[id])) / len(comp[id]))
        comp = comp.rename(columns={'fyearq': 'year', 'fqtr': 'quarter'})

        return comp, pred_id

    @staticmethod
    def load_data(load_original=False):

        if load_original:
            df = pd.read_stata('data/norman_data.dta')
            df.dtypes
            df.shape

            len(df['datadate'].unique())

            col = ['datadate', 'tic', 'ret_cc', 'mktcap', 'vol', 'indbvol', 'indbtrd', 'indsvol', 'indstrd', 'oidvol',
                   'turnover',
                   'sent_p', 'sent_n', 'sent_m', 'sentc_p', 'sentc_n', 'sentc_m',
                   'socsent_p', 'socsent_n', 'socsent_m', 'socsentc_p', 'socsentc_n', 'socsentc_m',
                   'socsent_p_TWT', 'socsent_p_STK', 'socsent_n_TWT', 'socsent_n_STK',
                   'socsent_m_TWT', 'socsent_m_STK', 'socsentc_p_TWT',
                   'socsentc_p_STK', 'socsentc_n_TWT', 'socsentc_n_STK',
                   'socsentc_m_TWT', 'socsentc_m_STK']

            df.head()
            df = df.filter(col)
            re_name_dict = {'datadate': 'date', 'ret_cc': 'ret'}
            df = df.rename(columns=re_name_dict)
            df.to_pickle('data/smaller_data.p')
        else:
            df = pd.read_pickle('data/smaller_data.p')
        #
        # df_analyst = pd.read_csv('data/analyst.csv')
        # df_analyst.date = pd.to_datetime(df_analyst.date, format='%Y%m%d')
        # df_analyst = df_analyst.iloc[:, 1:]
        #
        # df = df.merge(df_analyst)
        df.head()

        return df

    @staticmethod
    def load_ibes_long(reset=False, with_price_ret=False, with_compustat=False, with_news=False): # TODO develop the with variation if needed
        if reset:
            ibes = pd.read_csv('data/ibes_long_2.csv')
            ibes = ibes.rename(
                columns={'ANALYS': 'analyst', 'ANNDATS': 'andate', 'REVDATS': 'revdate', 'ACTDATS': 'acdats',
                         'OFTIC': 'tic', 'VALUE': 'value',
                         'FPEDATS': 'tgdate',
                         'ACTUAL': 'actual', 'FPI': 'fpi'})

            ibes['andate'] = pd.to_datetime(ibes['andate'], format='%Y%m%d')
            ibes['acdats'] = pd.to_datetime(ibes['acdats'], format='%Y%m%d')
            ibes['tgdate'] = pd.to_datetime(ibes['tgdate'], format='%Y%m%d')

            t = ibes[['tic', 'tgdate', 'actual']].drop_duplicates().sort_values(['tic', 'tgdate'])
            t['actual_L1'] = t.groupby('tic')['actual'].shift(1)
            t['actual_L2'] = t.groupby('tic')['actual'].shift(2)
            t['actual_L3'] = t.groupby('tic')['actual'].shift(3)
            t['actual_L4'] = t.groupby('tic')['actual'].shift(4)
            t = t.drop(columns=['actual'])
            ibes = ibes.merge(t, on=['tic', 'tgdate'], how='left')
            ibes = ibes.sort_values(['tic', 'tgdate'])
            if with_price_ret:
                ibes = Loader.add_quarterly_ret(ibes)
            if with_news:
                ibes = Loader.add_news_data(ibes)
            if with_compustat:
                # ibes = Loader.add_news_data(ibes)
                comp, comp_id = Loader.load_compustat()
                ibes['year'] = ibes['andate'].dt.year
                ibes['quarter'] = 1 + np.floor(ibes['andate'].dt.month / 4)
                ibes = ibes.merge(comp, on=['year', 'quarter', 'tic'], how='left')
                # ibes.to_pickle('data/ibes_processed.p')
            ibes.to_pickle('data/ibes_long_processed.p')
        else:
            ibes = pd.read_pickle('data/ibes_long_processed.p')
            ibes.head()

        return ibes

    @staticmethod
    def merge_results_df_with_price_actuals(df,reset=False, date_min=None,date_max=None, pred_std=['pred','consensus_mean','consensus_median']):
        permno_tic_translate = pd.read_csv('data/permno_tic.csv')
        permno_tic_translate = permno_tic_translate[['ticker', 'permno']]
        #start by loading the big ret number with the actual
        if reset:
            # General rule _s suffix means standardised (usually by tic and tgdate)
            ibes = Loader.load_ibes_long(reset=False)
            ibes = ibes[~pd.isnull(ibes['actual'])]

            # loading the translater saved before

            ibes = ibes.rename(columns={'TICKER': 'ticker'})
            # merging to keep only the one which were kept as a match with crsp
            ibes = ibes.merge(permno_tic_translate, on='ticker', how='inner')
            # load and clean crsp data
            ret = pd.read_csv('data/ibes_crsp.csv')
            ret.head()
            ret = ret.rename(columns={'PRC': 'price', 'VOL': 'vol', 'PERMNO': 'permno', 'RET': 'ret', 'NUMTRD': 'numtrd'})
            ret = ret.rename(columns={'date': 'andate'})
            ret['andate'] = pd.to_datetime(ret['andate'], format='%Y%m%d')

            ret['ret'] = pd.to_numeric(ret['ret'], 'coerc').fillna(0)

            # merge it
            ret.permno = ret.permno.astype('int64')

            ret = ret.merge(ibes, how='left', on=['permno', 'andate'])
            ret = ret.sort_values(['permno', 'andate'])

            ret = ret[['andate', 'permno', 'actual','price']]
            ret.to_pickle('data/ret_temp_f.p')
        else:
            ret = pd.read_pickle('data/ret_temp_f.p')

        if date_min is not None:
            ret = ret[ret['andate']>=date_min]
        if date_max is not None:
            ret = ret[ret['andate']<=date_max]


        df = df.merge(permno_tic_translate, how='left', right_on='ticker', left_on='tic')
        df = df[~pd.isnull(df['permno'])]

        df = df.drop(columns=['actual'])
        df = df.merge(ret, how='right', on=['permno', 'andate'])

        df = df.sort_values(['permno', 'andate'])

        ## adding the tg columns to have the target date for each
        tg = df[['permno', 'tgdate']].drop_duplicates()
        tg['tg'] = tg['tgdate']
        tg = tg.rename(columns={'tgdate': 'andate'})
        tg = tg[~pd.isnull(tg['tg'])]
        tg.head()
        df = df.merge(tg, how='outer', on=['permno', 'andate'])
        df = df.sort_values(['permno', 'andate'])
        df['tg'] = df.groupby(['permno'])['tg'].fillna(method='ffill')
        df['tg'] = df.groupby(['permno'])['tg'].fillna(method='bfill')

        # now we can expand the prediction by tg and permno
        def temp_expand_function(dataframe, colname):
            dataframe[colname] = dataframe.groupby(['permno', 'tg'])[colname].fillna(method='ffill')
            dataframe[colname] = dataframe.groupby(['permno', 'tg'])[colname].fillna(method='bfill')
            return dataframe

        df = temp_expand_function(dataframe=df, colname='actual')
        ret['actual'] = ret['actual'] / ret['price']

        for n in pred_std:
            df = temp_expand_function(dataframe=df, colname=n)
            # df[n] = df.groupby(['permno', 'tg'])[n].fillna(method='ffill')
            df[n+'_reg'] = (df[n]-df[n].mean())/df[n].std()
            df[n+'_error'] = (df[n]-df['actual']).abs()

        df = df[~pd.isnull(df[pred_std[0]])]
        df = df[~pd.isnull(df[pred_std[0]])]
        return df



    @staticmethod
    def load_ibes_with_feature(reset = False):
        if reset:
            # General rule _s suffix means standardised (usually by tic and tgdate)
            ibes = Loader.load_ibes_long(reset=False)
            ibes = ibes[~pd.isnull(ibes['actual'])]


            # loading the translater saved before
            permno_tic_translate = pd.read_csv('data/permno_tic.csv')
            permno_tic_translate = permno_tic_translate[['ticker','permno']]
            ibes = ibes.rename(columns={'TICKER':'ticker'})
            # merging to keep only the one which were kept as a match with crsp
            ibes=ibes.merge(permno_tic_translate,on='ticker',how='inner')
            # load and clean crsp data
            ret = pd.read_csv('data/ibes_crsp.csv')
            ret.head()
            ret = ret.rename(columns={'PRC':'price','VOL':'vol','PERMNO':'permno','RET':'ret','NUMTRD':'numtrd'})
            ret['date'] = pd.to_datetime(ret['date'],format='%Y%m%d')
            ret['ret'] =pd.to_numeric(ret['ret'],'coerc').fillna(0)
            ret = ret.sort_values(['permno','date'])
            ret['ret5']=ret.groupby('permno')['ret'].rolling(5).mean().reset_index()['ret']
            ret['ret10']=ret.groupby('permno')['ret'].rolling(10).mean().reset_index()['ret']
            ret['ret30']=ret.groupby('permno')['ret'].rolling(30).mean().reset_index()['ret']
            ret['ret60'] = ret.groupby('permno')['ret'].rolling(60).mean().reset_index()['ret']

            # merge it
            ret.permno = ret.permno.astype('int64')
            ret=ret.rename(columns={'date':'andate'})

            ibes[['permno','andate']].dtypes
            ibes= ibes.merge(ret,how='inner',on=['permno','andate'])

            ibes = ibes.sort_values(['tic', 'tgdate', 'andate'])


            # standardizing value and target value by price at time of recommendation!
            ibes['value'] = ibes['value']/ibes['price']
            ibes['actual'] = ibes['actual']/ibes['price']


            # creating the consensus measure for comparaisons
            ibes['consensus_mean'] = ibes.groupby(['tic', 'tgdate'])['value'].apply(lambda x: x.expanding().mean())
            ibes['consensus_median'] = ibes.groupby(['tic', 'tgdate'])['value'].apply(lambda x: x.expanding().median())
            # ibes['consensus_std'] = ibes.groupby(['tic', 'tgdate'])['value'].apply(lambda x: x.expanding().std())

            # first we remove not a time in andate and no tickers...
            ibes = ibes[~pd.isnull(ibes['andate'])]
            ibes = ibes[~pd.isnull(ibes.tic)]

            # just create a year column to latter on remove data easilly
            ibes['year'] = ibes.andate.dt.year
            # creating the perf measure, standardized perf, lag standardized perf
            ibes = ibes.sort_values(['tic', 'tgdate'])
            ibes['error'] = np.abs(ibes['actual'] - ibes['value'])
            ibes['mean_error'] = ibes.groupby(['tgdate', 'tic'])['error'].transform('mean')
            ibes['std_error'] = ibes.groupby(['tgdate', 'tic'])['error'].transform('std')
            ibes['error_s'] = ((ibes['error'] - ibes['mean_error']) / (ibes['std_error'])).clip(-5,5)
            ibes['analyst_mean_error_s'] = ibes.groupby(['tic', 'analyst', 'tgdate'])['error_s'].transform('mean')
            ibes['analyst_mean_error'] = ibes.groupby(['tic', 'analyst', 'tgdate'])['error'].transform('mean')
            ibes['analyst_mean_error_s_L1'] = ibes.groupby(['tic', 'analyst'])['analyst_mean_error_s'].transform(
                'shift')
            ibes['analyst_mean_error_L1'] = ibes.groupby(['tic', 'analyst'])['analyst_mean_error'].transform(
                'shift')
            ibes['analyst_mean_error_s_L2'] = ibes.groupby(['tic', 'analyst'])['analyst_mean_error_s'].transform(
                'shift', 2)
            ibes['analyst_mean_error_L2'] = ibes.groupby(['tic', 'analyst'])['analyst_mean_error'].transform(
                'shift', 2)
            ibes['analyst_std_error_s_L1'] = ibes.groupby(['tic', 'analyst'])['std_error'].transform(
                'shift')


            # herding and bold behavior plus abs dist to consensus

            # ibes['analyst_lag_value'] = ibes.groupby(['tic','analyst','tgdate'])['value'].transform('shift')
            ibes['analyst_lag_value'] = ibes.groupby(['tic','analyst'])['value'].transform('shift')
            ibes['change'] = ibes['value']-ibes['analyst_lag_value']
            ibes['sign_of_change'] = ibes['change'].apply(np.sign)
            ibes['abs_change'] = ibes['change'].apply(np.abs)



            ibes['bold'] = 1 * ((ibes['analyst_lag_value'] - ibes['consensus_mean']) > (ibes['value'] - ibes['consensus_mean']))
            ibes['herd'] = 1 * ((ibes['analyst_lag_value'] - ibes['consensus_mean']) < (ibes['value'] - ibes['consensus_mean']))
            ibes['abs_dist_to_consensus'] = (ibes['value']-ibes['consensus_mean']).abs()
            ibes['signed_dist_to_consensus'] = (ibes['value']-ibes['consensus_mean'])


            # the sign error
            ibes['error_signed'] = (ibes['actual'] - ibes['value'])
            ibes['error_signed_mean'] = ibes.groupby(['tic', 'analyst', 'tgdate'])['error_signed'].transform('mean')
            ibes['error_signed_mean_L1'] = ibes.groupby(['tic', 'analyst'])['error_signed'].transform('shift')
            ibes['error_signed_mean_L2'] = ibes.groupby(['tic', 'analyst'])['error_signed'].transform('shift', 2)

            # the number of firm followed
            ibes['nb_firm_followed_by_analyst'] = ibes.groupby(['analyst', 'tgdate'])['tic'].transform('nunique')
            ibes['nb_firm_pred_by_analyst'] = ibes.groupby(['analyst', 'tgdate', 'tic'])['value'].transform('count')

            # number of analyst who follow the firm
            ibes['nb_analyst_following_firm'] = ibes.groupby(['tgdate', 'tic'])['analyst'].transform('nunique')
            ibes['nb_analyst_following_firm'] = (ibes['nb_analyst_following_firm']-ibes['nb_analyst_following_firm'].mean())/ibes['nb_analyst_following_firm'].std()
            # number of prediction already made
            ibes = ibes.sort_values(['tic', 'tgdate', 'andate'])
            ibes['nb_pred_so_far'] = ibes.groupby(['tic', 'tgdate'])['value'].apply(lambda x: x.expanding().count())

            # number of pred total
            ibes['nb_pred_total'] = ibes.groupby(['tic', 'tgdate'])['value'].transform('count')
            ibes['nb_pred_total'] = (ibes['nb_pred_total']-ibes['nb_pred_total'].mean())/ibes['nb_pred_total'].std()

            # create the days dist betwen tgdate and andate and remove data with negative dates...
            ibes['days_to_actual'] = (ibes['tgdate'] - ibes['andate']).dt.days
            ibes = ibes[ibes['days_to_actual'] > 0]

            # firm experience so we first create the start date observed and then the experience.
            ibes['first_forecast_date_pre_tic'] = ibes.groupby(['tic', 'analyst'])['andate'].transform('min') # per tic, not pre tic...
            ibes['first_forecast_date_total'] = ibes.groupby(['analyst'])['andate'].transform('min')

            ibes['exp_firm'] = (ibes['andate'] - ibes['first_forecast_date_pre_tic']).dt.days
            ibes['exp_tot'] = (ibes['andate'] - ibes['first_forecast_date_total']).dt.days

            ibes['exp_firm_s'] = ibes.groupby(['tic', 'tgdate'])['exp_firm'].transform(
                lambda x: (x - x.mean()) / x.std())
            ibes['exp_tot_s'] = ibes.groupby(['tic', 'tgdate'])['exp_tot'].transform(lambda x: (x - x.mean()) / x.std())
            ibes['exp_tot_med'] = ibes.groupby(['tic', 'tgdate'])['exp_tot'].transform(lambda x: (x > x.median()) * 1)
            ibes['exp_firm_med'] = ibes.groupby(['tic', 'tgdate'])['exp_firm'].transform(lambda x: (x > x.median()) * 1)

            # replace inf with nan to be able to drop them easilly
            ibes = ibes.replace([np.inf, -np.inf], np.nan)

            # finaly we can save the ibes
            ibes = ibes.reset_index(drop=True)
            ibes = ibes.reset_index()
            ibes = ibes.rename(columns={'index': 'id'})
            ibes.actual.describe()

            # finaly we remove the very extreme ones!!!
            q=ibes['actual'].quantile([0.01,0.99])
            ibes.shape
            ibes = ibes[ibes['actual']>=q.iloc[0]]
            ibes = ibes[ibes['actual']<=q.iloc[1]]

            ibes.to_pickle('data/ibes_with_features.p')
            return ibes
        else:
            ibes = pd.read_pickle('data/ibes_with_features.p')
            return ibes


    feature = ['analyst_mean_error_s_L1', 'analyst_mean_error_s_L2', 'error_signed_mean_L1',
               'analyst_mean_error_L1','analyst_mean_error_L2',
            'error_signed_mean_L2', 'nb_pred_total','analyst_std_error_s_L1',
            'nb_firm_followed_by_analyst', 'nb_firm_pred_by_analyst', 'nb_pred_so_far', 'days_to_actual',
            'nb_analyst_following_firm',
            'exp_firm', 'exp_tot', 'exp_firm_s', 'exp_tot_s', 'exp_tot_med', 'exp_firm_med',
               'bold', 'herd', 'abs_dist_to_consensus', 'change', 'sign_of_change','signed_dist_to_consensus',
               'ret','ret5','ret10','ret30','ret60']

    feature_to_std = ['analyst_mean_error_s_L1', 'analyst_mean_error_s_L2', 'error_signed_mean_L1',
               'analyst_mean_error_L1','analyst_mean_error_L2',
            'analyst_std_error_s_L1',
            'nb_firm_followed_by_analyst', 'nb_firm_pred_by_analyst', 'nb_pred_so_far', 'days_to_actual',
            'exp_firm', 'exp_tot', 'exp_firm_s', 'exp_tot_s',
            'abs_dist_to_consensus', 'change','signed_dist_to_consensus','ret']
