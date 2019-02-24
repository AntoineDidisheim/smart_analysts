#####################################
# iclink                            #
# Link IBES (Ticker) & CRSP (Permno)#
# May 2018                          #
# Qingyi (Freda) Song Drechsler     #
#####################################

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import wrds

###################
# Connect to WRDS #
###################
conn=wrds.Connection()

###################
# IBES Part       #
###################
# IBES: Get the list of IBES Tickers for US firms in IBES
ibes1 = conn.raw_sql("""
                      select ticker, cusip, cname, sdates from ibes.idsum
                      where usfirm=1 and cusip != ''
                      """, date_cols=['sdates'])

# first and last 'start date' for a given cusip
ibes1_fdate = ibes1.groupby(['ticker','cusip'])['sdates'].min().reset_index()
ibes1_ldate = ibes1.groupby(['ticker','cusip'])['sdates'].max().reset_index()
ibes1_fdate = ibes1_fdate.rename(columns={'sdates':'fdate'})
ibes1_ldate = ibes1_ldate.rename(columns={'sdates':'ldate'})

ibes2 = pd.merge(ibes1, ibes1_fdate, on=['ticker','cusip'], how='left')
ibes2 = pd.merge(ibes2, ibes1_ldate, on=['ticker','cusip'], how='left')
ibes2 = ibes2.sort_values(by=['ticker','cusip','sdates'])

# keep only the most recent company name
ibes2 = ibes2[ibes2['sdates']==ibes2['ldate']].drop(['sdates'], axis=1)

###################
# CRSP Part       #
###################
# CRSP: Get all permno-ncusip combinations
crsp1 = conn.raw_sql("""
                      select distinct permno, ncusip, comnam, 
                      namedt, nameenddt
                      from crsp.stocknames
                      where ncusip != ''
                      """, date_cols=['namedt','nameenddt'])

# Given a permno-ncusip combo find the ealiest namedt and last nameenddt
crsp1_namedt = crsp1.groupby(['permno','ncusip'])['namedt'].min().reset_index()
crsp1_nameenddt = crsp1.groupby(['permno','ncusip'])['nameenddt'].max().reset_index()
crsp1_dtrange = pd.merge(crsp1_namedt, crsp1_nameenddt, \
                          on = ['permno','ncusip'], how='inner')

crsp1 = crsp1.drop(['namedt'],axis=1).rename(columns={'nameenddt':'enddt'})
crsp2 = pd.merge(crsp1, crsp1_dtrange, on =['permno','ncusip'], how='inner')

# keep only most recent company name
crsp2 = crsp2[crsp2['enddt']==crsp2['nameenddt']]
crsp2=crsp2.drop(['enddt'], axis=1)

###################
# Create Link     #
###################

## Option 1: Link by full cusip ##

link1_1 = pd.merge(ibes2, crsp2, how='inner', left_on='cusip', right_on='ncusip')
link1_1 = link1_1.sort_values(['ticker','permno','ldate'])

# Keep link with most recent company name
link1_1_tmp = link1_1.groupby(['ticker'])['ldate'].max().reset_index()
link1_2 = pd.merge(link1_1, link1_1_tmp, how='inner', on =['ticker','ldate'])

# Calculate distance of company names
# fuzzywuzzy - 100=highest match 0=lowest match
link1_2['name_similarity']=link1_2\
.apply(lambda row: fuzz.token_set_ratio(row['comnam'], row['cname']), axis=1)

# 10% percentile of the company name similarity
name_similarity_p10 = link1_2.name_similarity.quantile(0.10)

# Assign score for companies matched by full cusip and passing name distance
def score1(row):
    if (row['fdate']<=row['nameenddt']) & (row['ldate']>=row['namedt']) \
    & (row['name_similarity'] >= name_similarity_p10):
        value = 0
    elif (row['fdate']<=row['nameenddt']) & (row['ldate']>=row['namedt']):
        value=1
    elif row['name_similarity'] >= name_similarity_p10:
        value=2
    else:
        value=3
    return value

# assign size portfolio
link1_2['score']=link1_2.apply(score1, axis=1)
link1_2 = link1_2[['ticker','permno','cname','comnam','name_similarity','score']]
link1_2 = link1_2.drop_duplicates()

## Option 2: Link unmatched by comp.security table ibtic column ##
nomatch1 = pd.merge(ibes2[['ticker','cname']], link1_2[['permno','ticker']], on='ticker', how='left')
nomatch1 = nomatch1[nomatch1.permno.isnull()]
nomatch1 = nomatch1.drop(['permno'], axis=1).drop_duplicates()

# Use ccm table to add gvkey and ibtic
ccm=conn.raw_sql("""
                  select gvkey, liid as iid, lpermno as permno
                  from crsp.ccmxpf_linktable
                  where substr(linktype,1,1)='L'
                  and (linkprim ='C' or linkprim='P')
                  """)

# join with compustat security for ibtic
comp_sec = conn.raw_sql("""
                        select gvkey, iid, ibtic from comp.security
                        """)

ccmib = pd.merge(ccm, comp_sec, on=['gvkey','iid'], how='inner')
ccmib = ccmib[ccmib['ibtic'].notnull()].drop(['gvkey','iid'],axis=1)
ccmib = ccmib.drop_duplicates().rename(columns={'ibtic':'ticker'})

link2_1 = pd.merge(nomatch1, ccmib, how='inner', on='ticker')
link2_2=link2_1[link2_1['permno'].notnull()]

# add back crsp company names for final quality check
# pick latest comnam from crsp
crsp2_last = crsp2.groupby(['permno'])['nameenddt'].max().reset_index()
crsp3 = pd.merge(crsp2, crsp2_last, on=['permno', 'nameenddt'], how='inner')
link2_3 = pd.merge(link2_2, crsp3[['permno','comnam']], on=['permno'], how='left')

# calculate name similarity
link2_3['name_similarity']=link2_3\
.apply(lambda row: fuzz.token_set_ratio(row['comnam'],row['cname']), axis=1)
name_sim_p10 = link2_3.name_similarity.quantile(0.10)

link2_3['score']=np.where(link2_3['name_similarity']>=name_sim_p10, 0, 1)

# Combine the two sources for final output
iclink= link1_2.append(link2_3)
iclink = iclink[['ticker','permno','cname', 'comnam', 'name_similarity','score']]
iclink['permno']=iclink['permno'].astype(int)
iclink = iclink.drop_duplicates()

iclink.to_csv('data/icilink.csv')



############################# second part
from loader import Loader
pd.set_option('display.max_columns', 500)

iclink = pd.read_csv('data/icilink.csv')
ibes = Loader.load_ibes_long(reset=False)

ibes.andate.min()
iclink.head()
ibes.head()
ibes = ibes[['TICKER','fpi']].rename(columns={'TICKER':'ticker'})
ibes = ibes.reset_index(drop=True)
ibes = ibes.drop_duplicates()
df = iclink.merge(ibes, how='inner',on=['ticker'])

df = df[df.score==0]

df.to_csv('data/permno_tic.csv')

