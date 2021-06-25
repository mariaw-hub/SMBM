import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime
from pytz import timezone
import statsmodels.api as sm
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime as dt
from sklearn import linear_model

#%% Convert NDVI to KC values using linear regression model using dominant crop of the site 
#and create linear regression with max/min NDVI and max/min kc from FAO56 for crop 
# use either copr or airs file to create ndvi data, from NDVI folder 
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Python/NDVI/COPR_NDVI_normalized.csv" #new normalized data 27.1.21
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Python/NDVI/AIRS_NDVI_normalized.csv" #new normalized data 27.1.21


NDVI_data = pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0],sep=';') #from 2013-2019
       
#%% Centering technique to get average values after Roche et al., 

    
a= NDVI_data['median']#.rolling(window=3, center=True).mean() #centered moving average over 5 observations
idx=pd.date_range('01-01-2007','10-31-2019',freq='D') #daily daterange

a=a.reindex(idx) #upsampling to daily values 
a= a.interpolate() #interpolating monthly values to daily

a= a.loc['2007-10':]

ndvi_copr=pd.DataFrame(a)
ndvi_copr['Date']=pd.to_datetime(ndvi_copr.index)
ndvi_copr_m=a.resample('MS', loffset=pd.Timedelta(14,'d')).mean() #average mid-month start and end-month values 
#ndvi_copr_m=a.resample('M').median()
ndvi_copr_m=pd.DataFrame(ndvi_copr_m)
ndvi_copr_m['Site']='COPR' #monthly mid month average COPR

ndvi_airs=pd.DataFrame(a)
ndvi_airs['Date']=pd.to_datetime(ndvi_airs.index)
ndvi_airs_m=a.resample('MS', loffset=pd.Timedelta(14,'d')).mean() #average mid-month start and end-month values 
ndvi_airs_m=pd.DataFrame(ndvi_airs_m) #monthly mid month average COPR
ndvi_airs_m['Site']='AIRS'

COPR_PET=COPR_PET.loc['2007-10-13':]
COPR_PET_m=COPR_PET.resample('M').sum()



#%%

#COPR
file = ndvi_copr.join(COPR_PET['PET'], how='outer') #COPR
file=file.loc['2007-10-13':'2019-09-30']

#AIRS
file = ndvi_airs.join(Air_PET['PET'],how='outer') #AIRS
file=file.loc['2007-10-13':'2019-09-30']



a=file['median'].max()
b=file['median'].min()


kc_vi_copr = 1 - (a - file['median']) / (a - b) #replaces kc for AET estimation
kc_vi_copr.to_csv('kc_VI_copr.csv',index=True, sep=',') #COPR

#%%

kc_vi_airs = 1 - (a - ndvi_airs['median']) / (a - b) #replaces kc for AET estimation
kc_vi_airs.to_csv('kc_VI_airs.csv',index=True, sep=',') #AIRS



ax1 = (file['PET']*kc_vi_airs).loc['2010':'2013'].plot(color='orange')
ax2 = ax1.twinx()
ax2.spines['right'].set_position(('axes', 1.0))
file['median'].loc['2010':'2013'].plot(ax=ax2, color='green', linestyle='--')


















