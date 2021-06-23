
#Calculation of ET after Penman Montheith
import math 
import numpy as np
import pandas as pd
import pylab
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.stats import ks_2samp,mstats,kstest
import seaborn as sns
from datetime import datetime
import pytz
#%%##################################################################################

#Import Data set 
Air_data = pd.read_excel('AIRS_ET.xlsx')
Air_data.rename(columns={'TIMESTAMP': 'Date'}, inplace=True)

#Convert Date to Index
Air_data['Date'] = pd.to_datetime(Air_data['Date'])
Air_data = Air_data.set_index('Date')

# Convert UTC to PST
Air_data=Air_data.tz_localize('UTC')
Air_data=Air_data.tz_convert('US/Pacific')
Air_data = Air_data[(Air_data.index > '2007-10-01')]

Air_data[Air_data == 'NAN'] = np.nan
Air_data=Air_data.dropna()

for col in list(Air_data.columns):
        Air_data[col] = pd.to_numeric(Air_data[col])

#Air_data.drop(Air_data.columns[16:66], axis=1, inplace=True)

Air_data=Air_data.resample('D').mean()

#Air_data = Air_data.loc['2011-10-01':'2018-12-31'] #for CCS calc. 

#%%########################PENMAN MONTHEITH CALCULATION FOR PET##################
#%%
gradient=pd.DataFrame([2,2,2,2,3,4,5,6,6,6,6,4,2,2,2,2,3,4,5,6,6,6,6,4,2,2,2,2,3,4,5,6,6,6,6,4,2,2,2,2,3,4,5,6,6,6,6,4,2,2,2,2,3,4,5,6,6,6,6,4,2,2,2,
                       2,3,4,5,6,6,6,6,4,2,2,2,2,3,4,5,6,6,6,6,4,2,2,2])
idx=pd.date_range('10-01-2011','12-31-2018',freq='M')
gradient=gradient.set_index(pd.DatetimeIndex(idx)) #upsampling to daily values 
gradient['month']=gradient.index.month
gradient['Date'] = pd.to_datetime(gradient.index)

#%%########################PENMAN MONTHEITH CALCULATION FOR PET##################

AT1=pd.DataFrame(Air_data['AirTC_1_Avg'])
idx=pd.date_range('10-01-2011','12-31-2018',freq='D')
AT1=AT1.set_index(pd.DatetimeIndex(idx)) #upsampling to daily values 
AT1['Date'] = pd.to_datetime(AT1.index)
AT1 = pd.merge(AT1,gradient, left_on=AT1['Date'].apply(lambda x: (x.year, x.month)),
         right_on=gradient['Date'].apply(lambda y: (y.year, y.month)),
         how='outer')[['AirTC_1_Avg',0]]
AT1=AT1.set_index(pd.DatetimeIndex(Air_data.index)) #upsampling to daily values 

AT1['AT_new']=AT1['AirTC_1_Avg']+AT1[0]

AT2=pd.DataFrame(Air_data['AirTC_2_Avg'])
idx=pd.date_range('10-01-2011','12-31-2018',freq='D')
AT2=AT2.set_index(pd.DatetimeIndex(idx)) #upsampling to daily values 
AT2['Date'] = pd.to_datetime(AT2.index)
AT2 = pd.merge(AT2,gradient, left_on=AT2['Date'].apply(lambda x: (x.year, x.month)),
         right_on=gradient['Date'].apply(lambda y: (y.year, y.month)),
         how='outer')[['AirTC_2_Avg',0]]
AT2=AT2.set_index(pd.DatetimeIndex(Air_data.index)) #upsampling to daily values 

AT2['AT_new']=AT2['AirTC_2_Avg']+AT2[0]

AT1=AT1['AT_new']
AT2=AT2['AT_new']

#%%
#Saturation Vapour Pressure 
AT1=Air_data['AirTC_1_Avg'] 
#AT1 = (AT1.ffill()+AT1.bfill())/2
#AT1 = AT1.bfill().ffill()

AT2=Air_data['AirTC_2_Avg'] 
#AT2 = (AT2.ffill()+AT2.bfill())/2
#AT2 = AT2.bfill().ffill()
#%%

Es1=0.61078*np.exp(17.269*AT1/(AT1+237.3))
Es2=0.61078*np.exp(17.269*AT2/(AT2+237.3))

#Actual Vapour Pressure

RH2=Air_data['RH_avg_2']
#RH2=RH2.interpolate()

Ea1=(Es1 * RH2) / 100
Ea2=(Es2 * RH2) / 100

#Vapour Pressure Deficit 
VPD=Es1-Ea1 # used AT1 not AT2

#%%Net Radiation 
Rnet=Air_data['CNR_Wm2_Avg']

#%%Slope for vapour pressure curve
delta = 4098*(0.6108*np.exp((17.27*AT2)/(AT2+237.3)))/(AT2+237.3)**2 #ok

#%%Pressure
P=101.3*np.exp(-0.381/8.434) #ok

#Psychromatic Constant
pc=(1005*P)/(0.622*2468000) #ok

#%%Temperature in K at Height 1 and 2
TK1=AT1+273.16
TK2=AT2+273.16

#Density of air at height 2 (280cm)
R=287.05 #J/kgK #fixd for COPR
pa=P/(R*TK2)*1000 #ok
#%%Temperature at surface from longwave radiation
e=0.95 #Dars AIRS excel
sigma=0.0000000567 #Dars excel 

Tsurf=AT1*1.6984+264.11 # values replace the missing Lwup sensor which is needed for Tsurf calculation  normally 

#Temperature at Soil Depth 1 in K
ST1=Air_data['SoilTC_1_Avg']

Ts1= ST1+273.16 #ok

#Temperature difference between soil surface and depth 1
Tdiff=Ts1-Tsurf #ok

#%%Soil Heat Flux
K=0.39887137 #Dars excel
Dz=0.15 #Dars Excel 
G=Tdiff / Dz*K #ok

#%%delta ET
cp=1005 #(Jkg-1K-1) #fixed 
ch= 0.3 #for AIRS crop height changed 28.10 (decreased from 0.5)
zm=3.17 #fixed 
d=2/3*ch
zom=0.123*ch
zoh=0.1*ch
zh=2.85 #Fixed 
#%%
uz=Air_data['WS_ms_Avg'] #Windspeed at height 2
#uz = (uz.ffill()+uz.bfill())/2
#uz = uz.bfill().ffill()


LAI=24*ch*0.5
rl=200 #variable from Dars excel
rs=rl/LAI

ra=np.log((zm-d)/zom)*np.log((zh-d)/zoh) / 0.41**2
Ra=ra/uz
ts=900 #sampling interval - 15 min
LV=2468000 #Dar

#%%
LET = (delta * (Rnet + G) + pa * cp * (VPD / Ra)) / (delta + pc * (1 + rs / Ra))
#%%Bowen Ratio without the if statement
B_air = P * cp *(AT2-AT1) / (LV * 0.622 * (Ea2-Ea1))
LE_air = (Rnet + G)/(B_air+1)
ET_b_air = LE_air * 900 / LV
ET_b_air[ET_b_air < 0] = 0 
ET_b_air = ET_b_air.resample('D').sum()
plt.plot(ET_b_air)

#%% Reference ET from grass surface (ET0/PET) daily timestep

Rnet= Rnet.resample('D').sum() * 0.0009
G = G.resample('D').sum() * 0.0009
VPD = VPD.resample('D').mean()
u2 = uz.resample('D').mean()
AT2=AT2.resample('D').max()
delta = (uz.resample('D').mean())*4.87 / np.log(67.8*3.17-5.42)

PET_air1=(0.408*delta*(Rnet+G)+((pc*900*VPD*u2)/(AT2+273)))/(delta+pc*(1+0.34*u2))
PET_air[PET_air1 < 0] = 0 
plt.plot(PET_air1)

#%%dAILY CROP EVAPOTRANSPIRATION

PET_air=LET*86400/1000/LV*1000 #after Dars calculation (900 for 15 min, 9000 for 25 hours)

PET_air=pd.DataFrame(PET_air)
PET_air.columns=['PET']
PET_air['PET'][PET_air['PET'] < 0] = 0
#PET_air = PET_air.resample('D').sum()
PET_air=PET_air[["PET"]].astype(float).replace(0,np.nan)
#PET_all['Date'] = pd.to_datetime(PET_all.index)


PET_air=PET_air.interpolate(method='time')



#%%

PET_air.to_csv("RET_AIR_CCS.csv",sep=';',date_format='%Y-%m-%d')
#
#%%Infiltration
I_AIR=pd.concat([PET_air, Air_Rain], axis=1)
#I_AIR = I_AIR.resample('D').sum()
I_AIR.to_csv("Infiltration_AIR180221.csv", sep='\t')



