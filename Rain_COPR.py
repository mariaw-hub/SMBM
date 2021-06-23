import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gamma
import datetime as dt
import matplotlib.dates as mdates
from scipy.stats import genextreme
#%%
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
#%%
rain = np.asarray(COPR_data['Rain_mm_Tot'].values, dtype=np.float64)
date = np.asarray(COPR_data.index, dtype = np.datetime64)

#%% Identify start and end of rain events.
rain_boolean = (rain > 0.254).astype(int)
diff = np.diff(rain_boolean)
start = (diff == +1)
end = (diff == -1)
#%%
start = np.nonzero(start)[0]
end = np.nonzero(end)[0]
nevents = len(start)
#%%
rain_per_event = np.empty(nevents)
min_per_event = np.empty(nevents)
max_per_event = np.empty(nevents)
mean_per_event = np.empty(nevents)
start_date_event = np.empty(nevents, dtype='datetime64[us]')
end_date_event = np.empty(nevents, dtype='datetime64[us]')
time_between_events = np.empty(nevents, dtype = 'datetime64[us]')
#%%
for i in range(nevents):
    this_start_index = start[i]
    this_end_index = end[i]
    
    start_date_event[i] = (date[this_start_index+1])
    end_date_event[i] = (date[this_end_index+1])
       
    this_rain = rain[this_start_index+1:this_end_index+1] 
    rain_per_event[i] = np.sum(this_rain) #storm total per event
    min_per_event[i] = this_rain.min()
    max_per_event[i] = this_rain.max()
    mean_per_event[i] = this_rain.mean()
    
#%%  combine rain per event with datetime as index 
ST = pd.DataFrame(rain_per_event, start_date_event)
ST.columns=['ST']

#%%
ST11=pd.concat([ST.loc['2010-01-26':'2012-02-14'],ST.loc['2019-02-19': ]])
ST11['Period']= 'ND'

#No Drought - AIRS
ST21=pd.concat([ST_air.loc['2010-01-26':'2012-02-14'],ST_air.loc['2019-02-19': ]])
ST21['Period']= 'ND1'

#Moderate Drought - COPR
ST12=pd.concat([ST.loc['2008-01-29':'2009-04-01'],ST.loc['2009-10-27':'2010-01-26' ], 
                ST.loc['2012-02-14':'2013-04-30'], ST.loc['2017-03-01':'2018-01-23']])
ST12['Period']= 'MD'

#Moderate Drought - AIRS
ST22=pd.concat([ST_air.loc['2008-01-29':'2009-04-01'],ST_air.loc['2009-10-27':'2010-01-26' ], 
                ST_air.loc['2012-02-14':'2013-04-30'], ST_air.loc['2017-03-01':'2018-01-23']])
ST22['Period']= 'MD1'

#Extreme Drought - COPR
ST13=pd.concat([ST.loc['2013-05-01':'2017-03-01'],ST.loc['2018-01-23':'2019-02-19'],
                ST.loc['2007-05-01':'2008-01-29']])
ST13['Period']= 'ED'

#Extreme Drought - AIRS
ST23=pd.concat([ST_air.loc['2013-05-01':'2017-03-01'],ST_air.loc['2018-01-23':'2019-02-19'],
                ST_air.loc['2007-05-01':'2008-01-29']])
ST23['Period']= 'ED1'
 
Storm_Totals= pd.concat([ST11,ST12,ST13,ST21,ST22,ST23], axis =0)
#%%




