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
COPR_data = pd.read_excel(r'/Users/mariawarter/Box Sync/PhD/PhD/Python/Data/COPR_climate.xlsx')
COPR_data.rename(columns={'TIMESTAMP': 'Date'}, inplace=True)

#Convert Date to Index
COPR_data=COPR_data.set_index(pd.DatetimeIndex(COPR_data['Date'])) 

# Convert UTC to PST
COPR_data=COPR_data.tz_localize('UTC')
COPR_data=COPR_data.tz_convert('US/Pacific')
COPR_data = COPR_data[(COPR_data.index > '2007-10-01')] #changed to include WY 2008 for P

COPR_data[COPR_data == 'NAN'] = np.nan

for col in list(COPR_data.columns):
        COPR_data[col] = pd.to_numeric(COPR_data[col])
COPR_data['Date'] = pd.to_datetime(COPR_data['Date'])
#%% Import PET from ET dataframe
COPR_PET = pd.read_csv(r'/Users/mariawarter/Box Sync/PhD/PhD/Python/Data/RET_COPR_03032021.csv',dayfirst=True,parse_dates=True,sep=';',index_col=(0))
COPR_PET.index=pd.to_datetime(COPR_PET.index)
COPR_PET['Date'] = pd.to_datetime(COPR_PET.index)

COPR_PET = COPR_PET.resample('M').sum() 
#COPR_PET.loc['2017-10-31'] = 61.8904

#COPR_PET = COPR_PET.mask(COPR_PET['PET'].between(0, 0.01))
#COPR_PET = (COPR_PET.ffill()+COPR_PET.bfill())/2
#COPR_PET = COPR_PET.bfill().ffill()
COPR_PET['Site']='COPR'
#%%################AIRSTRIP#####################################################

Air_data = pd.read_excel(r'/Users/mariawarter/Box Sync/PhD/PhD/Python/Data/AIRS_climate2.xlsx')

Air_data.rename(columns={'TIMESTAMP': 'Date'}, inplace=True)

#Convert Date to Index
Air_data=Air_data.set_index(pd.DatetimeIndex(Air_data['Date'])) 

# Convert UTC to PST
Air_data=Air_data.tz_localize('UTC')
Air_data=Air_data.tz_convert('US/Pacific')
Air_data = Air_data[(Air_data.index > '2007-10-01')]

Air_data[Air_data == 'NAN'] = np.nan
 
for col in list(Air_data.columns):
        Air_data[col] = pd.to_numeric(Air_data[col])

Air_data['Date'] = pd.to_datetime(Air_data['Date'])

#Air_data.drop(Air_data.columns[20:71], axis=1, inplace=True)
#%%
Air_PET = pd.read_csv(r'/Users/mariawarter/Box Sync/PhD/PhD/Python/Data/RET_AIR.csv', sep=';',dayfirst=True,parse_dates=True,index_col=(0))
Air_PET.index=pd.to_datetime(Air_PET.index)
Air_PET['Date'] = pd.to_datetime(Air_PET.index)
Air_PET= Air_PET.resample('M').sum()


#Air_PET = Air_PET.mask(Air_PET['PET'].between(0, 0.01))
#Air_PET = (Air_PET.ffill()+Air_PET.bfill())/2
#Air_PET = Air_PET.bfill().ffill()


Air_PET['Site']= 'AIRS'

#%% Convert Data to Water year starting in October 2007 (=WY 2008) earliest start possible. 
COPR_Rain = COPR_data[['Date', 'Rain_mm_Tot']]
COPR_Rain = COPR_Rain.loc['2007-10-01':'2019-09-30']
COPR_Rain = COPR_Rain.resample('M').sum()

#COPR_Rain_M = COPR_Rain.resample('M').sum()

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
#COPR_SM2 = COPR_data[['SMwfv_2_Avg']]
COPR_SM1.columns = ['SM 1']
#COPR_SM2.columns = ['SM 2']

COPR_SM1 = COPR_SM1.loc['2007-10-01':]
#Monthly mean doesnt have water year 2
COPR_SM1_M = COPR_SM1.resample('M').mean()
#COPR_SM2_M = COPR_SM2.resample('M').mean()

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
Air_Rain = Air_Rain.resample('M').sum()
Air_Rain = Air_Rain.loc['2007-10-01': '2019-09-30']

Air_Rain['Date'] = pd.to_datetime(Air_Rain.index)
Air_Rain['Month']=Air_Rain.index.month

def assign_wy(Air_Rain):
    if Air_Rain.Date.month>=10:
        return(pd.datetime(Air_Rain.Date.year+1,1,1).year)
    else:
        return(pd.datetime(Air_Rain.Date.year,1,1).year)

Air_Rain['WY'] = Air_Rain.apply(lambda x: assign_wy(x), axis=1)
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

por2=0.340
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
#Air_RH = Air_RH.loc['2008-01-01':'2019-09-30']
Air_RH['Site'] = 'AIRS'

#%%Separation into the 3 Drought periods 
#Temperature 
#No Drought - COPR
at11=pd.concat([COPR_AT.loc['2010-01-26':'2012-02-14'],COPR_AT.loc['2019-02-19': ]])
at11['Period']= 'ND'

#No Drought - AIRS
at21=pd.concat([Air_AT.loc['2010-01-26':'2012-02-14'],Air_AT.loc['2019-02-19': ]])
at21['Period']= 'ND1'

#Moderate Drought - COPR
at12=pd.concat([COPR_AT.loc['2008-01-29':'2009-04-01'],COPR_AT.loc['2009-10-27':'2010-01-26' ], 
                COPR_AT.loc['2012-02-14':'2013-04-30'], COPR_AT.loc['2017-03-01':'2018-01-23']])
at12['Period']= 'MD'

#Moderate Drought - AIRS
at22=pd.concat([Air_AT.loc['2008-01-29':'2009-04-01'],Air_AT.loc['2009-10-27':'2010-01-26' ], 
                Air_AT.loc['2012-02-14':'2013-04-30'], Air_AT.loc['2017-03-01':'2018-01-23']])
at22['Period']= 'MD1'

#Extreme Drought - COPR
at13=pd.concat([COPR_AT.loc['2013-05-01':'2017-03-01'],COPR_AT.loc['2018-01-23':'2019-02-19'],
                COPR_AT.loc['2007-05-01':'2008-01-29']])
at13['Period']= 'ED'

#Extreme Drought - AIRS
at23=pd.concat([Air_AT.loc['2013-05-01':'2017-03-01'],Air_AT.loc['2018-01-23':'2019-02-19'],
                Air_AT.loc['2007-05-01':'2008-01-29']])
at23['Period']= 'ED1'

AT= pd.concat([at11,at12,at13,at21,at22,at23], axis =0)
#%%Relative Humidity 
rh11=pd.concat([COPR_RH.loc['2010-01-26':'2012-02-14'],COPR_RH.loc['2019-02-19': ]])
rh11['Period']= 'ND'

#No Drought - AIRS
rh21=pd.concat([Air_RH.loc['2010-01-26':'2012-02-14'],Air_RH.loc['2019-02-19': ]])
rh21['Period']= 'ND1'

#Moderate Drought - COPR
rh12=pd.concat([COPR_RH.loc['2008-01-29':'2009-04-01'],COPR_RH.loc['2009-10-27':'2010-01-26' ], 
                COPR_RH.loc['2012-02-14':'2013-04-30'], COPR_RH.loc['2017-03-01':'2018-01-23']])
rh12['Period']= 'MD'

#Moderate Drought - AIRS
rh22=pd.concat([Air_RH.loc['2008-01-29':'2009-04-01'],Air_RH.loc['2009-10-27':'2010-01-26' ], 
                Air_RH.loc['2012-02-14':'2013-04-30'], Air_RH.loc['2017-03-01':'2018-01-23']])
rh22['Period']= 'MD1'

#Extreme Drought - COPR
rh13=pd.concat([COPR_RH.loc['2013-05-01':'2017-03-01'],COPR_RH.loc['2018-01-23':'2019-02-19'],
                COPR_RH.loc['2007-05-01':'2008-01-29']])
rh13['Period']= 'ED'

#Extreme Drought - AIRS
rh23=pd.concat([Air_RH.loc['2013-05-01':'2017-03-01'],Air_RH.loc['2018-01-23':'2019-02-19'],
                Air_RH.loc['2007-05-01':'2008-01-29']])
rh23['Period']= 'ED1'

RH = pd.concat([rh11,rh12,rh13,rh21,rh22,rh23], axis =0)
#%%potential ET
pet11=pd.concat([COPR_PET.loc['2010-01-26':'2012-02-14'],COPR_PET.loc['2019-02-19': ]])
pet11['Period']= 'ND'

#No Drought - AIRS
pet21=pd.concat([Air_PET.loc['2010-01-26':'2012-02-14'],Air_PET.loc['2019-02-19': ]])
pet21['Period']= 'ND1'

#Moderate Drought - COPR
pet12=pd.concat([COPR_PET.loc['2008-01-29':'2009-04-01'],COPR_PET.loc['2009-10-27':'2010-01-26' ], 
                COPR_PET.loc['2012-02-14':'2013-04-30'], COPR_PET.loc['2017-03-01':'2018-01-23']])
pet12['Period']= 'MD'

#Moderate Drought - AIRS
pet22=pd.concat([Air_PET.loc['2008-01-29':'2009-04-01'],Air_PET.loc['2009-10-27':'2010-01-26' ], 
                Air_PET.loc['2012-02-14':'2013-04-30'], Air_PET.loc['2017-03-01':'2018-01-23']])
pet22['Period']= 'MD1'

#Extreme Drought - COPR
pet13=pd.concat([COPR_PET.loc['2013-05-01':'2017-03-01'],COPR_PET.loc['2018-01-23':'2019-02-19'],
                COPR_PET.loc['2007-05-01':'2008-01-29']])
pet13['Period']= 'ED'

#Extreme Drought - AIRS
pet23=pd.concat([Air_PET.loc['2013-05-01':'2017-03-01'],Air_PET.loc['2018-01-23':'2019-02-19'],
                Air_PET.loc['2007-05-01':'2008-01-29']])
pet23['Period']= 'ED1'

PET = pd.concat([pet11,pet12,pet13,pet21,pet22,pet23], axis=0)
#%%
rain11=pd.concat([COPR_Rain.loc['2010-01-26':'2012-02-14'],COPR_Rain.loc['2019-02-19': ]])
rain11['Period']= 'ND'

#No Drought - AIRS
rain21=pd.concat([Air_Rain.loc['2010-01-26':'2012-02-14'],Air_Rain.loc['2019-02-19': ]])
rain21['Period']= 'ND1'

#Moderate Drought - COPR
rain12=pd.concat([COPR_Rain.loc['2008-01-29':'2009-04-01'],COPR_Rain.loc['2009-10-27':'2010-01-26' ], 
                COPR_Rain.loc['2012-02-14':'2013-04-30'], COPR_Rain.loc['2017-03-01':'2018-01-23']])
rain12['Period']= 'MD'

#Moderate Drought - AIRS
rain22=pd.concat([Air_Rain.loc['2008-01-29':'2009-04-01'],Air_Rain.loc['2009-10-27':'2010-01-26' ], 
                Air_Rain.loc['2012-02-14':'2013-04-30'], Air_Rain.loc['2017-03-01':'2018-01-23']])
rain22['Period']= 'MD1'

#Extreme Drought - COPR
rain13=pd.concat([COPR_Rain.loc['2013-05-01':'2017-03-01'],COPR_Rain.loc['2018-01-23':'2019-02-19'],
                COPR_Rain.loc['2007-05-01':'2008-01-29']])
rain13['Period']= 'ED'

#Extreme Drought - AIRS
rain23=pd.concat([Air_Rain.loc['2013-05-01':'2017-03-01'],Air_Rain.loc['2018-01-23':'2019-02-19'],
                Air_Rain.loc['2007-05-01':'2008-01-29']])
rain23['Period']= 'ED1'

#Rain = pd.concat([rain11,rain12,rain13,rain21,rain22,rain23], axis =0)

#%%P , T, RH, PET in one plot - Violin Plot for Climate 
#sns.set(style="whitegrid")
fig = plt.figure()
ax=fig.add_subplot(221)
ax = sns.violinplot(x="Period", y="AT2", showmedians=True, data=AT,linewidth = 0.3,hue='Site', dodge=False)
ax.legend(frameon=False,fontsize='small')
ax1 = fig.add_subplot(222)
ax1 = sns.violinplot(x='Period', y='RH',showmedians=True,linewidth = 0.3,  data=RH, hue='Site', dodge=False)
ax1.legend(frameon=False,fontsize='small')

ax2 = fig.add_subplot(223)
ax2 = sns.violinplot(x='Period', y='PET',showmedians=True, linewidth = 0.3,data=PET,cut=0, hue='Site', dodge=False)
ax2.legend(frameon=False,fontsize='small')

ax3 = fig.add_subplot(224)
ax3 = sns.violinplot(x='Period', y='Rain_mm_Tot',showmedians=True,linewidth = 0.3, cut=0, data=Rain, hue='Site', dodge=False)
ax3.legend(frameon=False,fontsize='small')

#%%

#dry_nov = Air_AT.assign(x=Air_AT.index.strftime('%m-%d')) \
              #.query("'11-01' <= x <= '11-30'").drop('x',1)
              
#dry_dec = Air_AT.assign(x=Air_AT.index.strftime('%m-%d')) \
              #.query("'12-01' <= x <= '12-31'").drop('x',1)             
              
#%% SM and NDVI in one plot, plus SM/ NDVI regression (extra)             

sm11=pd.concat([COPR_SM1_M.loc['2010-01-26':'2012-02-14'],COPR_SM1_M.loc['2019-02-19': ]])
sm11['Period']= 'ND'

#No Drought - AIRS
sm21=pd.concat([Air_SM1_M.loc['2010-01-26':'2012-02-14'],Air_SM1_M.loc['2019-02-19': ]])
sm21['Period']= 'ND1'

#Moderate Drought - COPR
sm12=pd.concat([COPR_SM1_M.loc['2008-01-29':'2009-04-01'],COPR_SM1_M.loc['2009-10-27':'2010-01-26' ], 
                COPR_SM1_M.loc['2012-02-14':'2013-04-30'], COPR_SM1_M.loc['2017-03-01':'2018-01-23']])
sm12['Period']= 'MD'

#Moderate Drought - AIRS
sm22=pd.concat([Air_SM1_M.loc['2008-01-29':'2009-04-01'],Air_SM1_M.loc['2009-10-27':'2010-01-26' ], 
                Air_SM1_M.loc['2012-02-14':'2013-04-30'], Air_SM1_M.loc['2017-03-01':'2018-01-23']])
sm22['Period']= 'MD1'

#Extreme Drought - COPR
sm13=pd.concat([COPR_SM1_M.loc['2013-05-01':'2017-03-01'],COPR_SM1_M.loc['2018-01-23':'2019-02-19'],
                COPR_SM1_M.loc['2007-05-01':'2008-01-29']])
sm13['Period']= 'ED'

#Extreme Drought - AIRS
sm23=pd.concat([Air_SM1_M.loc['2013-05-01':'2017-03-01'],Air_SM1_M.loc['2018-01-23':'2019-02-19'],
                Air_SM1_M.loc['2007-05-01':'2008-01-29']])
sm23['Period']= 'ED1'
SM = pd.concat([sm11,sm12,sm13,sm21,sm22,sm23], axis=0)
#%%
ndvi11 = pd.concat([ndvi_copr.loc['2010-01-26':'2012-02-14'],ndvi_copr.loc['2019-02-19': ]])
ndvi11['Period']= 'ND'

#No Drought - AIRS
ndvi21=pd.concat([ndvi_airs.loc['2010-01-26':'2012-02-14'],ndvi_airs.loc['2019-02-19': ]])
ndvi21['Period']= 'ND1'

#Moderate Drought - COPR
ndvi12=pd.concat([ndvi_copr.loc['2008-01-29':'2009-04-01'],ndvi_copr.loc['2009-10-27':'2010-01-26' ], 
                ndvi_copr.loc['2012-02-14':'2013-04-30'], ndvi_copr.loc['2017-03-01':'2018-01-23']])
ndvi12['Period']= 'MD'

#Moderate Drought - AIRS
ndvi22=pd.concat([ndvi_airs.loc['2008-01-29':'2009-04-01'],ndvi_airs.loc['2009-10-27':'2010-01-26' ], 
                ndvi_airs.loc['2012-02-14':'2013-04-30'], ndvi_airs.loc['2017-03-01':'2018-01-23']])
ndvi22['Period']= 'MD1'

#Extreme Drought - COPR
ndvi13=pd.concat([ndvi_copr.loc['2013-05-01':'2017-03-01'],ndvi_copr.loc['2018-01-23':'2019-02-19'],
                ndvi_copr.loc['2007-05-01':'2008-01-29']])
ndvi13['Period']= 'ED'

#Extreme Drought - AIRS
ndvi23=pd.concat([ndvi_airs.loc['2013-05-01':'2017-03-01'],ndvi_airs.loc['2018-01-23':'2019-02-19'],
                ndvi_airs.loc['2007-05-01':'2008-01-29']])
ndvi23['Period']= 'ED1'


NDVI = pd.concat([ndvi11,ndvi12,ndvi13,ndvi21,ndvi22,ndvi23], axis=0)
#%% Violinplot for SM and NDVI for PD and D
         
fig=plt.figure()
ax=fig.add_subplot(211)
ax=sns.violinplot(x='Period', y='Saturation', data=SM,linewidth = 0.3,hue='Site', dodge=False)              
ax.legend(frameon=False, fontsize='small')            
ax1=fig.add_subplot(212)
ax1=sns.violinplot(x='Period', y='median', data=NDVI, linewidth = 0.3, hue='Site', dodge=False)
ax1.legend(frameon=False, fontsize='small')            
              
#%% Violinplot for Net P

Net_P = pd.read_csv("Infiltration_COPR180221.csv", sep=';') #COPR
Net_P = pd.read_csv("Infiltration_AIR180221.csv", sep='\t') #AIRS

Net_P['Date'] = pd.to_datetime(Net_P['Date'], utc=True)
Net_P=Net_P.set_index(pd.DatetimeIndex(Net_P['Date'])) 
Net_P = Net_P.resample('D').sum()
Net_P['Date'] = pd.to_datetime(Net_P.index)
Net_P = Net_P.loc['2007-10-01':]

def assign_wy(Net_P):
    if Net_P.Date.month>=10:
        return(pd.datetime(Net_P.Date.year+1,1,1).year)
    else:
        return(pd.datetime(Net_P.Date.year,1,1).year)

Net_P['WY'] = Net_P.apply(lambda x: assign_wy(x), axis=1)
Net_P['diff'] = (Net_P['Rain_mm_Tot'] - Net_P['PET']).groupby(Net_P['WY']).cumsum()

Net_P['Site'] = 'COPR'
Net_P_copr=Net_P

Net_P['Site'] = 'AIRS'
Net_P_airs=Net_P
#%%
netP11 = pd.concat([Net_P_copr.loc['2010-01-26':'2012-02-14'],Net_P_copr.loc['2019-02-19': ]])
netP11['Period']= 'ND'

#No Drought - AIRS
netP21=pd.concat([Net_P_airs.loc['2010-01-26':'2012-02-14'],Net_P_airs.loc['2019-02-19': ]])
netP21['Period']= 'ND1'

#Moderate Drought - COPR
netP12=pd.concat([Net_P_copr.loc['2008-01-29':'2009-04-01'],Net_P_copr.loc['2009-10-27':'2010-01-26' ], 
                Net_P_copr.loc['2012-02-14':'2013-04-30'], Net_P_copr.loc['2017-03-01':'2018-01-23']])
netP12['Period']= 'MD'

#Moderate Drought - AIRS
netP22=pd.concat([Net_P_airs.loc['2008-01-29':'2009-04-01'],Net_P_airs.loc['2009-10-27':'2010-01-26' ], 
                Net_P_airs.loc['2012-02-14':'2013-04-30'], Net_P_airs.loc['2017-03-01':'2018-01-23']])
netP22['Period']= 'MD1'

#Extreme Drought - COPR
netP13=pd.concat([Net_P_copr.loc['2013-05-01':'2017-03-01'],Net_P_copr.loc['2018-01-23':'2019-02-19'],
                 Net_P_copr.loc['2007-05-01':'2008-01-29']])
netP13['Period']= 'ED'

#Extreme Drought - AIRS
netP23=pd.concat([Net_P_airs.loc['2013-05-01':'2017-03-01'],Net_P_airs.loc['2018-01-23':'2019-02-19'],
                Net_P_airs.loc['2007-05-01':'2008-01-29']])
netP23['Period']= 'ED1'

NETP = pd.concat([netP11,netP12,netP13,netP21,netP22,netP23], axis=0)
#%%Violinplot of available P

ax=sns.violinplot(x='Period', y='diff', data=NETP,linewidth = 0.3, hue='Site', dodge=False) 
             
#%% LIneplot for the differen NDVI years separately 
n0=np.asarray(ndvi_copr_m['median'].loc['2008']) #MD
n1=np.asarray(ndvi_copr_m['median'].loc['2011']) #ND
n2=np.asarray(ndvi_copr_m['median'].loc['2015']) #ED
n3=np.asarray(ndvi_copr_m['median'].loc['2016']) #ED
#n6=np.asarray(ndvi_copr_m['median'].loc['2019']) #ND

na0=np.asarray(ndvi_airs_m['median'].loc['2008'])
na1=np.asarray(ndvi_airs_m['median'].loc['2011'])
na2=np.asarray(ndvi_airs_m['median'].loc['2015'])
na3=np.asarray(ndvi_airs_m['median'].loc['2016'])
#na6=np.asarray(ndvi_airs_m['median'].loc['2019'])

x_labels=['J','F','M','A','M','J','J','A','S','O','N','D']
x=np.array([0,1,2,3,4,5,6,7,8,9,10,11])
plt.style.use('bmh')
plt.subplot(211)
plt.plot(n0, label='2008')
plt.plot(n1, label='2011')
plt.plot(n2, label='2015', linestyle='--')
plt.plot(n3,label='2016', linestyle='--')
#plt.plot(n6,label='2019')
plt.ylim(0.15,0.8)
plt.xticks(x,x_labels,fontsize='medium', fontweight='bold')
plt.yticks(fontsize='medium', fontweight='bold')
plt.ylabel('NDVI', fontweight='bold',fontsize='medium')
plt.axhline(0.3,color='black',linewidth=0.5)

plt.legend()   
plt.subplot(212)
plt.plot(na0, label='2008')
plt.plot(na1, label='2011')
plt.plot(na2, label='2015',linestyle='--')
plt.plot(na3,label='2016',linestyle='--')
#plt.plot(na6,label='2019')
plt.legend()
plt.ylim(0.15,0.8)
plt.yticks(fontsize='medium', fontweight='bold')
plt.xticks(x,x_labels,fontsize='medium', fontweight='bold')
plt.ylabel('NDVI', fontweight='bold',fontsize='medium')
plt.axhline(0.3,color='black',linewidth=0.5)

#%% Regression plot SM and NDVI with browning threshold

x=np.asarray(COPR_SM1_M['SM 1']).reshape(-1,1)
y=np.asarray(ndvi_copr_m['median'].loc[:'2019-05']).reshape(-1,1)
x1=np.asarray(Air_SM1_M['SM 1'].dropna()).reshape(-1,1)
y1=np.asarray(ndvi_airs_m['median'].loc[(ndvi_airs_m.index < '2016-08-01') | (ndvi_airs_m.index > '2018-02-01')]).reshape(-1,1)

reg=LinearRegression().fit(x,y)
reg_pred=reg.predict(x)
reg.score(x,y)
ax=reg.coef_
bx=reg.intercept_
cx=(0.3-bx)/ax

reg1=LinearRegression().fit(x1,y1)
reg_pred1=reg.predict(x1)
reg.score(x1,y1)
ax1=reg1.coef_
bx1=reg1.intercept_
cx1=(0.3-bx1)/ax1

#%%Scatter Plot with regression of SM and NDVI using browning threshold
x1=np.asarray(Air_SM1_M['SM 1'].dropna())
y1=np.asarray(ndvi_airs_m['median'].loc[(ndvi_airs_m.index < '2016-08-01') | (ndvi_airs_m.index > '2018-02-01')])

slope, intercept, r_value, p_value, std_err = stats.linregress(x1,y1)

plt.scatter(x,y,color='blue')
plt.plot(x,reg_pred, color='red')
plt.scatter(x1,y1,color='orange')
sns.regplot(x1, y1,color='orange',
      ci=None, label="y={0:.4f}x+{1:.4f}".format(slope, intercept))
plt.axhline(0.3,color='black')
plt.axvline(cx,color='black')
plt.axvline(cx1,color='black')
plt.ylabel('NDVI', font='medium',fontweight='bold')
plt.xlabel('VMC (m $\mathregular{^3}$/ m $\mathregular{^3}$)',fontweight='bold', fontsize='medium' )
#%%

r=COPR_Rain_M['Rain_mm_Tot'].loc['2017-10':'2018-03']













     