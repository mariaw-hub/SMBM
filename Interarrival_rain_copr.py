#%% IMPORTS
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#%% IMPORT DATA
    # Read in historical COPR data from edited Excel file - Python/Data
# Import Data set
    
#COPR_data = pd.read_excel('COPR_climate2.xlsx')
#COPR_data.rename(columns={'TIMESTAMP': 'Date'}, inplace=True)

#Convert Date to Index
#COPR_data=COPR_data.set_index(pd.DatetimeIndex(COPR_data['Date'])) 

# Convert UTC to PST
#COPR_data=COPR_data.tz_localize('UTC')
#COPR_data=COPR_data.tz_convert('US/Pacific')
#COPR_data = COPR_data[(COPR_data.index > '2008-01-01')]

#COPR_data[COPR_data == 'NAN'] = np.nan

#for col in list(COPR_data.columns):
#        COPR_data[col] = pd.to_numeric(COPR_data[col])
#COPR_data['Date'] = pd.to_datetime(COPR_data['Date'])

#%% MAIN
    # Copy datetimes and rainfall totals to new dataframe
    rain = pd.DataFrame({'DT': COPR_data.Date.copy(),\
                         'TOTAL': COPR_data['Rain_mm_Tot'].copy()})
    rain = rain.resample('D').sum()
     # Convert column datatypes  
    #rain[rain == 'NAN'] = np.nan

    for col in list(rain.columns):
        rain[col] = pd.to_numeric(rain[col])
    
    rain['DT'] = pd.to_datetime(rain.index)
    rain = rain.dropna()
    rain = rain.loc[~(rain['TOTAL'] ==0.254)]
    rain = rain.loc[~(rain['TOTAL'] ==0.508)]
    rain = rain.loc[~(rain['TOTAL'] ==0.762)]
    rain = rain.loc[~(rain['TOTAL'] ==1.016)]
    rain = rain.loc[~(rain['TOTAL'] ==1.27)]
    rain = rain.loc[~(rain['TOTAL'] ==1.524)]
    rain = rain.loc[~(rain['TOTAL'] ==1.778)]


#%%    # Add column with 1 for rain event, 0 for no rain.
    rain.loc[rain.TOTAL != 0, 'RAIN'] = 1
    rain.loc[rain.TOTAL == 0, 'RAIN'] = 0
    
    # Calculate difference between binary rows to identify start and end of
    # each event
    rain['DIFF'] = rain.RAIN.diff()
    rain = rain.set_index(rain.DT)
    
    # Get indices of start of each rainfall event as list
    start = list(rain[rain.DIFF == 1].index)
    start_date = pd.DataFrame(rain[rain.DIFF == 1].index)
    start_date['DT'] = pd.to_datetime(start_date['DT'])
    # Get indices of end of each rainfall event as list
    end = list(rain[rain.DIFF == -1].index)
    end_date = pd.DataFrame(rain[rain.DIFF == -1].index)
    
#%%    # EVENT DURAITON
    # Initialize empty lists for duration of each event and time between events.
    dur_list = []
    btw_list = []
    # Iterate through lists with start and end indices
    for i,(s,e) in enumerate(zip(start,end)):
        # Add column with each consecutive event numbered
        rain.loc[s:e, 'EVENT'] = i + 1
        # Calculate duration of each event
        t_dur = rain.DT[e] - rain.DT[s]
        if i == 0:
            # For the first event, calculate time since beginning of data (Not
            # really useful, since actual time since last event is unknown.)
            t_btw = rain.DT[s] - rain.DT[0]
        else:
            # For all other events, calculate time since previous event.
            t_btw = rain.DT[s] - rain.DT[end[i-1]]
        # Add duration of event and time since previous event to respective lists
        dur_list.append(t_dur)
        btw_list.append(t_btw)

#%%    # Add event #, duration, and time since last event to new dataframe
    events = pd.DataFrame({'EVENT': list(range(1,len(start)+1)),\
                           'DUR': dur_list,\
                           'BTW': btw_list})
    
    events['DUR_min'] = events.DUR.dt.seconds / 60
    events['DUR_hrs'] = events.DUR_min / 60
    events['BTW_min'] = events.BTW.dt.days * 24 *60 + events.BTW.dt.seconds/60 
    events['BTW_hrs'] = events.BTW_min / 60
    events['BTW_days'] = (events.BTW_hrs / 24).round()
    
    
    events = events.set_index(pd.DatetimeIndex(start_date['DT']))
    
    # Add Storm totals to df - ST df from Rain script
    #events = pd.concat([events, ST], axis=1, sort=False)
    events['Month']=events.index.month
    #events['Intens'] = events['ST'] / events['DUR_hrs']
    events['Date'] = pd.to_datetime(events.index)
    events = events.loc['2008-01-01':'2019-09-30']
    events['DOY'] = events['Date'].dt.dayofyear
    
#%%  get column with water year on the events df  
def assign_wy(events):
    if events.Date.month>=10:
        return(pd.datetime(events.Date.year+1,1,1).year)
    else:
        return(pd.datetime(events.Date.year,1,1).year)

events['WY'] = events.apply(lambda x: assign_wy(x), axis=1)

#%%Splitting Events df into the drought periods 
    
 events_pd = events.loc['2008-01-01' : '2011-12-31'] 
 events_d = events.loc['2012-01-01' : '2018-12-31']
#%%Storm Totals

j=np.asarray(events_pd['ST'][events_pd['ST'] > 2]) #pre drought
k=np.asarray(events_d['ST'][events_d['ST'] > 2]) #drought
#%%Duration 
l=np.asarray(events_pd['DUR_min'][events_pd['DUR_min'] > 30]) #pre drought
m=np.asarray(events_d['DUR_min'][events_d['DUR_min'] > 30]) #drought

#%%Interarrival times
n=np.asarray(events_pd['BTW_hrs'][events_pd['BTW_hrs'] > 24]) #pre-drought
p=np.asarray(events_d['BTW_hrs'][events_d['BTW_hrs'] > 24]) #drought
#%% Intensity (ST/DUR) in mm/hr
s= np.asarray(events_pd['Intens'][events_pd['Intens'] > 2])
r= np.asarray(events_d['Intens'][events_d['Intens'] > 2])

#%% Stats
#Comparison of Storm  Totals > 2
 stats.ks_2samp(j,k) # ks=0.053, p value 0.98 as of 17.09.2019
 
#Comparison of Storm Duration > 60 min
 stats.ks_2samp(l,m) #, stat=0.07, p value =0.84 as of 17.09.2019
 
#Comparison of Interarrival times > 60 min 
 stats.ks_2samp(n,p) #ks=0.116, p-value=0.026 as of 17.09.2019
 
 #Intensity (ST/DUR) in mm/hr > 2mm/hour
  stats.ks_2samp(s,r) #ks=0.116, p-value=0.026 as of 17.09.2019

#%%
stat, p = stats.kruskal(COPR_Rain_pd['Rain_mm_Tot'],COPR_Rain_d['Rain_mm_Tot'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')

#%%

events_btw = pd.DataFrame(events[events['BTW_days'] > 31])
events_btw['Date'] = pd.to_datetime(events_btw.index)
events_btw['DOY'] = events_btw['Date'].dt.dayofyear
events_btw['DOY_s'] = events_btw['Date'] - events_btw['BTW']
events_btw['DOY_sd'] = events_btw['DOY_s'].dt.dayofyear


events_btw = events_btw[['DOY_s', 'DOY_sd']].copy()
events_btw = events_btw.set_index(pd.DatetimeIndex(events_btw['DOY_s']))
events_btw = events_btw.assign(x=events_btw.index.strftime('%m-%d')) \
             .query("'03-31' <= x <= '09-31'").drop('x',1)

events_pd = events_btw.loc['2008' : '2011'] 
events_pd['Period'] = 'PD'
events_pd['Period'] = events_pd['Period'].astype(str)

events_d = events_btw.loc['2012' : '2018']
events_d['Period'] = 'D'
events_d['Period'] = events_d['Period'].astype(str)

BTW_copr= pd.concat([events_pd,events_d])
BTW_copr = BTW_copr.loc['2008':'2017-06']

#%%
sns.boxplot(x='DOY_sd', y='Period',data=BTW_copr, color = 'blue')








