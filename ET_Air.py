
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
Air_data = Air_data[(Air_data.index > '2007-12-01')]

Air_data[Air_data == 'NAN'] = np.nan

for col in list(Air_data.columns):
        Air_data[col] = pd.to_numeric(Air_data[col])

#Air_data.drop(Air_data.columns[16:66], axis=1, inplace=True)

Air_Rain = Air_data['Rain_mm_Tot'].resample('D').sum()
Air_Rain = Air_Rain.loc['2007-12-01':'2019-09-30']

#%%########################PENMAN MONTHEITH CALCULATION FOR PET##################

#Saturation Vapour Pressure 
AT1=Air_data['AirTC_1_Avg'] + 2.0
#AT1 = (AT1.ffill()+AT1.bfill())/2
#AT1 = AT1.bfill().ffill()

AT2=Air_data['AirTC_2_Avg'] + 2.0
#AT2 = (AT2.ffill()+AT2.bfill())/2
#AT2 = AT2.bfill().ffill()


Es1=0.61078*np.exp(17.269*AT1/(AT1+237.3))
Es2=0.61078*np.exp(17.269*AT2/(AT2+237.3))

#Actual Vapour Pressure
RH1=Air_data['RH_avg_1']
#RH1 = (RH1.ffill()+RH1.bfill())/2
#RH1 = RH1.bfill().ffill()

RH2=Air_data['RH_avg_2']
#RH2 = (RH2.ffill()+RH2.bfill())/2
#RH2 = RH2.bfill().ffill()

Ea1=(Es1 * RH1) / 100
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
PET_air=LET*ts/1000/LV*1000 #after Dars calculation 
PET_air=pd.DataFrame(PET_air)
PET_air.columns=['PET']
PET_air['PET'][PET_air['PET'] < 0] = 0
PET_air = PET_air.resample('D').sum()
#PET_air_M = PET_air['PET'].resample('M').sum()
#PET_air = PET_air.loc['2007-12-01':'2019-09-30']

#%%

PET_air.to_csv("ET_AIR_CC.csv",sep='\t')
#
#%%Infiltration
I_AIR=pd.concat([PET_air, Air_Rain], axis=1)
#I_AIR = I_AIR.resample('D').sum()
I_AIR.to_csv("Infiltration_AIR020620.csv", sep='\t')



