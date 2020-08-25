import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime
from pytz import timezone
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress

#%%#####################################COPR#####################################
# Import Data set
COPR_data = pd.read_excel('COPR_climate2.xlsx')
COPR_data.rename(columns={'TIMESTAMP': 'Date'}, inplace=True)

#Convert Date to Index
COPR_data=COPR_data.set_index(pd.DatetimeIndex(COPR_data['Date'])) 

# Convert UTC to PST
COPR_data=COPR_data.tz_localize('UTC')
COPR_data=COPR_data.tz_convert('US/Pacific')
COPR_data = COPR_data[(COPR_data.index > '2008-01-01')]

COPR_data[COPR_data == 'NAN'] = np.nan

for col in list(COPR_data.columns):
        COPR_data[col] = pd.to_numeric(COPR_data[col])
COPR_data['Date'] = pd.to_datetime(COPR_data['Date'])
#%% Import PET from ET dataframe
COPR_PET = pd.read_csv('ET_COPR.csv', sep='\t')
COPR_PET['Date'] = pd.to_datetime(COPR_PET['Date'], utc=True)
COPR_PET=COPR_PET.set_index(pd.DatetimeIndex(COPR_PET['Date'])) 
COPR_PET = COPR_PET.resample('M').sum() 
COPR_PET.loc['2017-10-31'] = 61.8904

#COPR_PET = COPR_PET.mask(COPR_PET['PET'].between(0, 0.01))
#COPR_PET = (COPR_PET.ffill()+COPR_PET.bfill())/2
#COPR_PET = COPR_PET.bfill().ffill()
COPR_PET = COPR_PET.loc['2008-01-01' : ]
COPR_PET['Site']='COPR'
#%%################AIRSTRIP#####################################################

Air_data = pd.read_excel('AIRS_climate2.xlsx')

Air_data.rename(columns={'TIMESTAMP': 'Date'}, inplace=True)

#Convert Date to Index
Air_data=Air_data.set_index(pd.DatetimeIndex(Air_data['Date'])) 

# Convert UTC to PST
Air_data=Air_data.tz_localize('UTC')
Air_data=Air_data.tz_convert('US/Pacific')
Air_data = Air_data[(Air_data.index > '2008-01-01')]

Air_data[Air_data == 'NAN'] = np.nan
 
for col in list(Air_data.columns):
        Air_data[col] = pd.to_numeric(Air_data[col])

Air_data['Date'] = pd.to_datetime(Air_data['Date'])

#Air_data.drop(Air_data.columns[20:71], axis=1, inplace=True)
#%%
Air_PET = pd.read_csv('ET_AIR.csv', sep='\t')
Air_PET['Date'] = pd.to_datetime(Air_PET['Date'], utc=True)
Air_PET=Air_PET.set_index(pd.DatetimeIndex(Air_PET['Date'])) 
Air_PET = Air_PET.resample('M').sum()
Air_PET = Air_PET.mask(Air_PET['PET'].between(0, 0.01))
Air_PET = (Air_PET.ffill()+Air_PET.bfill())/2
Air_PET = Air_PET.bfill().ffill()

Air_PET = Air_PET.loc['2008-01-01' : ]
Air_PET['Site']= 'AIRS'

#%% Convert Data to Water year starting in October 2007 (=WY 2008) earliest start possible. 
COPR_Rain = COPR_data[['Date', 'Rain_mm_Tot']]
COPR_Rain = COPR_Rain.loc['2008-01-01':'2019-09-30']
COPR_Rain = COPR_Rain.resample('D').sum()

COPR_Rain['Date'] = pd.to_datetime(COPR_Rain.index)
COPR_Rain['Month'] = COPR_Rain.index.month

def assign_wy(COPR_Rain):
    if COPR_Rain.Date.month>=10:
        return(pd.datetime(COPR_Rain.Date.year+1,1,1).year)
    else:
        return(pd.datetime(COPR_Rain.Date.year,1,1).year)

COPR_Rain['WY'] = COPR_Rain.apply(lambda x: assign_wy(x), axis=1)
#COPR_Rain = COPR_Rain.loc['2008-10-01' : '2019-09-30']
COPR_Rain['Site'] = 'COPR'
#%%Get annual P totals for every WY
COPR_Rain_Y = pd.DataFrame(COPR_Rain['Rain_mm_Tot'].groupby(COPR_Rain['WY']).sum())
COPR_Rain_Y['Site'] = 'COPR'

#%% Soil Moisture ALL - is not by water year 
COPR_SM1= COPR_data[['SMwfv_1_Avg']]
COPR_SM1.columns = ['SM 1']
COPR_SM1 = COPR_SM1.loc['2008-01-01':]
#Monthly mean doesnt have water year 2
COPR_SM1_M = COPR_SM1.resample('M').mean()
#COPR_SM1_D = COPR_SM1.resample('D').mean()
COPR_SM1_M = (COPR_SM1_M.ffill()+COPR_SM1_M.bfill())/2
COPR_SM1_M = COPR_SM1_M.bfill().ffill()
#COPR_SM1_M['Date'] = pd.to_datetime(COPR_SM1_M.index)


#%% % Saturation of Moisture 
por1=0.710
COPR_SM1_M['Saturation']=COPR_SM1_M['SM 1']/por1
wp = 0.13
COPR_SM1_M['PAW']=COPR_SM1_M['SM 1'] - wp
COPR_SM1_M['Site'] = 'COPR'


#%%new DF with just the Temperatures of Air and Soil and RHs
COPR_T=COPR_data[[ 'AirTC_1_Max', 'AirTC_1_Min','AirTC_2_Max', 'AirTC_2_Min', 'RH_avg_2']]
#Get Daytime values for T and RH 
COPR_T=COPR_T.between_time('07:00','19:00')
#%% Get monthly averages for T according to WY 
COPR_AT=COPR_T[[ 'AirTC_2_Max']]
COPR_AT=COPR_AT.resample('D').max()
COPR_AT.columns=['AT2']
COPR_AT = (COPR_AT.ffill()+COPR_AT.bfill())/2
COPR_AT = COPR_AT.bfill().ffill()
COPR_AT['Date'] = pd.to_datetime(COPR_AT.index)
COPR_AT['Site'] = 'COPR'

#%%Relative Humidity 
COPR_RH = COPR_T[['RH_avg_2']]
COPR_RH = COPR_RH.resample('D').mean()
COPR_RH.columns = ['RH']
COPR_RH = (COPR_RH.ffill()+COPR_RH.bfill())/2
COPR_RH = COPR_RH.bfill().ffill()
COPR_RH['Date'] = pd.to_datetime(COPR_RH.index)
COPR_RH = COPR_RH.loc['2008-01-01':'2019-09-30']
COPR_RH['Site'] = 'COPR'


#%%Convert to Water year starting October 2008
Air_Rain = Air_data[['Date', 'Rain_mm_Tot']]
Air_Rain = Air_Rain.resample('D').sum()

Air_Rain['Date'] = pd.to_datetime(Air_Rain.index)
Air_Rain['Month']=Air_Rain.index.month

def assign_wy(Air_Rain):
    if Air_Rain.Date.month>=10:
        return(pd.datetime(Air_Rain.Date.year+1,1,1).year)
    else:
        return(pd.datetime(Air_Rain.Date.year,1,1).year)

Air_Rain['WY'] = Air_Rain.apply(lambda x: assign_wy(x), axis=1)
#Air_Rain = Air_Rain.loc['2008-10-01': '2019-09-30']
Air_Rain['Site']='AIRS'

#%%
Air_Rain_Y = pd.DataFrame(Air_Rain['Rain_mm_Tot'].groupby(Air_Rain['WY']).sum())
Air_Rain_Y['Site'] ='AIRS'

#%% DF for SM is not by water year 
Air_SM1= Air_data[['SMwfv_1_Avg']]
Air_SM1.columns=['SM 1']
#Air_SM1 = Air_SM1.loc['2008-01-01':'2019-09-30']
#Monthly mean doesnt have water year in it
Air_SM1_M = Air_SM1.resample('M').mean()
#Air_SM1_D = Air_SM1.resample('D').mean()

Air_SM1_M['Date'] = pd.to_datetime(Air_SM1_M.index)
Air_SM1_M['Site'] = 'AIRS'
#%% Saturation of Soil Moisture in %

por1=0.340
Air_SM1_M['Saturation']= Air_SM1_M['SM 1']/por1
wp=0.07
Air_SM1_M['PAW']= Air_SM1_M['SM 1'] - wp

#%%TEMPERATURE AND HUMIDITY DAYTIME
Air_T=Air_data[[ 'AirTC_1_Avg', 'AirTC_2_Avg','AirTC_2_Min', 'AirTC_2_Max', 'RH_avg_2']]

Air_T=Air_T.between_time('07:00','19:00')
#%%
Air_AT=Air_T[[ 'AirTC_2_Max']]
Air_AT.columns=['AT2']
Air_AT=Air_AT.resample('D').max()
Air_AT = Air_AT.fillna(Air_AT.mean())
Air_AT = (Air_AT.ffill()+Air_AT.bfill())/2
Air_AT = Air_AT.bfill().ffill()
Air_AT['Date'] = pd.to_datetime(Air_AT.index)
Air_AT['Site'] = 'AIRS'

#%%Relative Humidity 
Air_RH = Air_T[['RH_avg_2']]
Air_RH = Air_RH.resample('D').mean()
Air_RH.columns = ['RH']
Air_RH = (Air_RH.ffill()+Air_RH.bfill())/2
Air_RH = Air_RH.bfill().ffill()
Air_RH['Date'] = pd.to_datetime(Air_RH.index)
Air_RH = Air_RH.loc['2008-01-01':'2019-09-30']
Air_RH['Site'] = 'AIRS'

#%%
at1=COPR_AT.loc['2008-01-01':'2011-12-31']
at1['Period']= 'PD'
at2=COPR_AT.loc['2012-01-01':'2018-12-31']
at2['Period']= 'D'

stats.mannwhitneyu(at1['AT2'],at2['AT2'])
stats.ks_2samp(at1['AT2'],at2['AT2'])

at3=Air_AT.loc['2008-01-01':'2011-12-31']
at3['Period']= 'PD1'
at4=Air_AT.loc['2012-01-01':'2018-12-31']
at4['Period']= 'D1'
AT= pd.concat([at1,at2,at3,at4], axis=0)
#%%
rh1=COPR_RH.loc['2008-01-01':'2011-12-31'] 
rh1['Period']= 'PD'
rh2=COPR_RH.loc['2012-01-01':'2018-12-31'] 
rh2['Period']= 'D'

rh3=Air_RH.loc['2008-01-01':'2011-12-31'] 
rh3['Period']= 'PD1'
rh4=Air_RH.loc['2012-01-01':'2018-12-31'] 
rh4['Period']= 'D1'

RH = pd.concat([rh1,rh2,rh3,rh4], axis =0)
#%%
pet1=COPR_PET.loc['2008-01-01':'2011-12-31']
pet1['Period']= 'PD'
pet2=COPR_PET.loc['2012-01-01':'2018-12-31']
pet2['Period']= 'D'

pet3=Air_PET.loc['2008-01-01':'2011-12-31']
pet3['Period']= 'PD1'
pet4=Air_PET.loc['2012-01-01':'2018-12-31']
pet4['Period']= 'D1'
PET = pd.concat([pet1,pet2,pet3,pet4], axis=0)
#%%
rain1=COPR_Rain.loc['2008-01-01':'2011-12-31']
rain1['Period']= 'PD'
rain2=COPR_Rain.loc['2012-01-01':'2018-12-31']
rain2['Period']= 'D'

rain3=Air_Rain.loc['2008-01-01':'2011-12-31']
rain3['Period']= 'PD1'
rain4=Air_Rain.loc['2012-01-01':'2018-12-31']
rain4['Period']= 'D1'

Rain = pd.concat([rain1,rain2,rain3,rain4], axis =0)

#%%P , T, RH, PET in one plot 
#sns.set(style="whitegrid")
fig = plt.figure()
ax=fig.add_subplot(221)
ax = sns.violinplot(x="Period", y="AT2", showmedians=True, data=AT,linewidth = 0.3,hue='Site', dodge=False)

ax1 = fig.add_subplot(222)
ax1 = sns.violinplot(x='Period', y='RH',linewidth = 0.3,  data=RH, hue='Site', dodge=False)

ax2 = fig.add_subplot(223)
ax2 = sns.violinplot(x='Period', y='PET', linewidth = 0.3,data=PET,cut=0, hue='Site', dodge=False)

ax3 = fig.add_subplot(224)
ax3 = sns.violinplot(x='Period', y='Rain_mm_Tot',linewidth = 0.3, cut=0, data=Rain, hue='Site', dodge=False)

#%%

#dry_nov = Air_AT.assign(x=Air_AT.index.strftime('%m-%d')) \
              #.query("'11-01' <= x <= '11-30'").drop('x',1)
              
#dry_dec = Air_AT.assign(x=Air_AT.index.strftime('%m-%d')) \
              #.query("'12-01' <= x <= '12-31'").drop('x',1)             
              
#%% SM and NDVI in one plot, plus SM/ NDVI regression (extra)             

sm1=COPR_SM1_M.loc['2008-01-01':'2011-12-31']
sm1['Period']= 'PD'
sm2=COPR_SM1_M.loc['2012-01-01':'2018-12-31']
sm2['Period']= 'D'

sm3=Air_SM1_M.loc['2008-01-01':'2011-12-31']
sm3['Period']= 'PD1'
sm4=Air_SM1_M.loc['2012-01-01':'2018-12-31']
sm4['Period']= 'D1'
SM = pd.concat([sm1,sm2,sm3,sm4], axis=0)
#%%
ndvi1 = NDVI_copr.loc['2008-01-01':'2011-12-31']
ndvi1['Period'] ='PD'

ndvi2=NDVI_copr.loc['2012-01-01':'2018-12-31']
ndvi2['Period'] ='D'

ndvi3 = NDVI_airs.loc['2008-01-01':'2011-12-31']
ndvi3['Period'] ='PD1'

ndvi4=NDVI_airs.loc['2012-01-01':'2018-12-31']
ndvi4['Period'] ='D1'

NDVI = pd.concat([ndvi1,ndvi2,ndvi3,ndvi4], axis=0)
#%% Violinplot for SM and NDVI for PD and D
         
fig=plt.figure()
ax=fig.add_subplot(211)
ax=sns.violinplot(x='Period', y='Saturation', data=SM,linewidth = 0.3, hue='Site', dodge=False)              
              
ax1=fig.add_subplot(212)
ax1=sns.violinplot(x='Period', y='median', data=NDVI, linewidth = 0.3, hue='Site', dodge=False)              
#%% Violinplot for Net P

Net_P = pd.read_csv("Infiltration_COPR070420.csv", sep=',') #COPR
Net_P = pd.read_csv("Infiltration_AIR070420.csv", sep='\t') #AIRS

Net_P['Date'] = pd.to_datetime(Net_P['Date'], utc=True)
Net_P=Net_P.set_index(pd.DatetimeIndex(Net_P['Date'])) 
Net_P = Net_P.resample('M').sum()
Net_P['Date'] = pd.to_datetime(Net_P.index)
Net_P = Net_P.loc['2007-10-01':]

def assign_wy(Net_P):
    if Net_P.Date.month>=10:
        return(pd.datetime(Net_P.Date.year+1,1,1).year)
    else:
        return(pd.datetime(Net_P.Date.year,1,1).year)

Net_P['WY'] = Net_P.apply(lambda x: assign_wy(x), axis=1)
Net_P['diff'] = (Net_P['Rain_mm_Tot'] - Net_P['PET']).groupby(Net_P['WY']).cumsum()
#Net_P['Site'] = 'COPR'
Net_P['Site'] = 'AIRS'
#%%
netp1 = Net_P.loc['2008-01-01':'2011-12-31'] #COPR
netp1['Period'] ='PD' #COPR

netp2=Net_P.loc['2012-01-01':'2018-12-31'] 
netp2['Period'] ='D' 

netp3 = Net_P.loc['2008-01-01':'2011-12-31']#AIRS
netp3['Period'] ='PD1'

netp4=Net_P.loc['2012-01-01':'2018-12-31']
netp4['Period'] ='D1'

NETP = pd.concat([netp1,netp2,netp3,netp4], axis=0)
#%%

ax=sns.violinplot(x='Period', y='diff', data=NETP,linewidth = 0.3, hue='Site', dodge=False)              
#%%
n0=np.asarray(NDVI_copr['median'].loc['2010'])
n1=np.asarray(NDVI_copr['median'].loc['2011'])
n2=np.asarray(NDVI_copr['median'].loc['2015'])
n3=np.asarray(NDVI_copr['median'].loc['2016'])
n6=np.asarray(NDVI_copr['median'].loc['2019'])

na0=np.asarray(NDVI_airs['median'].loc['2010'])
na1=np.asarray(NDVI_airs['median'].loc['2011'])
na2=np.asarray(NDVI_airs['median'].loc['2015'])
na3=np.asarray(NDVI_airs['median'].loc['2016'])
na6=np.asarray(NDVI_airs['median'].loc['2019'])

plt.style.use('bmh')
plt.subplot(211)
plt.plot(n0, label='2010')
plt.plot(n1, label='2011')
plt.plot(n2, label='2015', linestyle='--')
plt.plot(n3,label='2016', linestyle='--')
plt.plot(n6,label='2019')
plt.ylim(0.15,0.8)
plt.yticks(fontsize='medium', fontweight='bold')
plt.xticks(fontsize='medium', fontweight='bold')

plt.legend()   
plt.subplot(212)
plt.plot(na0, label='2010')
plt.plot(na1, label='2011')
plt.plot(na2, label='2015',linestyle='--')
plt.plot(na3,label='2016',linestyle='--')
plt.plot(na6,label='2019')
plt.legend()
plt.ylim(0.15,0.8)
plt.yticks(fontsize='medium', fontweight='bold')
plt.xticks(fontsize='medium', fontweight='bold')

#%%

T1 = COPR_AT.assign(x=COPR_AT.index.strftime('%m-%d')) \
             .query("'03-01' <= x <= '10-31'").drop('x',1)

T1pd = T1['AT2'].loc['2008':'2011']
T1d = T1['AT2'].loc['2012':'2018']

T2 = Air_AT.assign(x=Air_AT.index.strftime('%m-%d')) \
             .query("'03-01' <= x <= '10-31'").drop('x',1)

T2pd = T2['AT2'].loc['2008':'2011']
T2d = T2['AT2'].loc['2012':'2018']














     