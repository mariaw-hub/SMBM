
#Calculation of ET after Penman Montheith
import math 
import numpy as np
import pandas as pd
import pylab
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pytz

#%%##################################################################################

#Import Data set 
COPR_data = pd.read_excel('COPR_ET.xlsx')
COPR_data.rename(columns={'TIMESTAMP': 'Date'}, inplace=True)

#Convert Date to Index
COPR_data=COPR_data.set_index(pd.DatetimeIndex(COPR_data['Date'])) 
COPR_data['Date'] = pd.to_datetime(COPR_data.index)

# Convert UTC to PST
COPR_data=COPR_data.tz_localize('UTC')
COPR_data=COPR_data.tz_convert('US/Pacific')
#COPR_data = COPR_data[(COPR_data.index > '2007-06-01')]
#COPR_data = COPR_data.loc['2011-10-01':'2018-12-31'] #for CCS calc. 

COPR_data =COPR_data.loc['2007-10-01':]
COPR_data[COPR_data == 'NAN'] = np.nan

for col in list(COPR_data.columns):
        COPR_data[col] = pd.to_numeric(COPR_data[col])
COPR_data['Date'] = pd.to_datetime(COPR_data['Date'])

COPR_data = COPR_data.resample('D').mean()

COPR_data=COPR_data.interpolate()

#for CCS scenarios cut data to 2012-2018

COPR_data = COPR_data.loc['2011-10-01':'2018-12-31'] #for CCS calc. 


#%%Rain

COPR_Rain = COPR_data['Rain_mm_Tot'].resample('D').sum()
#COPR_Rain = COPR_Rain.loc['2007-10-01':'2019-09-30']

#%%
gradient=pd.DataFrame([2,2,2,2,3,4,5,6,6,6,6,4,2,2,2,2,3,4,5,6,6,6,6,4,2,2,2,2,3,4,5,6,6,6,6,4,2,2,2,2,3,4,5,6,6,6,6,4,2,2,2,2,3,4,5,6,6,6,6,4,2,2,2,
                       2,3,4,5,6,6,6,6,4,2,2,2,2,3,4,5,6,6,6,6,4,2,2,2])
idx=pd.date_range('10-01-2011','12-31-2018',freq='M')
gradient=gradient.set_index(pd.DatetimeIndex(idx)) #upsampling to daily values 
gradient['month']=gradient.index.month
gradient['Date'] = pd.to_datetime(gradient.index)

#%%########################PENMAN MONTHEITH CALCULATION FOR PET##################

AT1=pd.DataFrame(COPR_data['AirTC_1_Avg'])
idx=pd.date_range('10-01-2011','12-31-2018',freq='D')
AT1=AT1.set_index(pd.DatetimeIndex(idx)) #upsampling to daily values 
AT1['Date'] = pd.to_datetime(AT1.index)
AT1 = pd.merge(AT1,gradient, left_on=AT1['Date'].apply(lambda x: (x.year, x.month)),
         right_on=gradient['Date'].apply(lambda y: (y.year, y.month)),
         how='outer')[['AirTC_1_Avg',0]]
AT1=AT1.set_index(pd.DatetimeIndex(COPR_data.index)) #upsampling to daily values 

AT1['AT_new']=AT1['AirTC_1_Avg']+AT1[0]

AT2=pd.DataFrame(COPR_data['AirTC_2_Avg'])
idx=pd.date_range('10-01-2011','12-31-2018',freq='D')
AT2=AT2.set_index(pd.DatetimeIndex(idx)) #upsampling to daily values 
AT2['Date'] = pd.to_datetime(AT2.index)
AT2 = pd.merge(AT2,gradient, left_on=AT2['Date'].apply(lambda x: (x.year, x.month)),
         right_on=gradient['Date'].apply(lambda y: (y.year, y.month)),
         how='outer')[['AirTC_2_Avg',0]]
AT2=AT2.set_index(pd.DatetimeIndex(COPR_data.index)) #upsampling to daily values 

AT2['AT_new']=AT2['AirTC_2_Avg']+AT2[0]

AT1=AT1['AT_new']
AT2=AT2['AT_new']

#%%

AT1=COPR_data['AirTC_1_Avg']
AT2=COPR_data['AirTC_2_Avg']
#Saturation Vapour Pressure 
#%%
Es1=0.61078*np.exp(17.269*AT1/(AT1+237.3))
Es2=0.61078*np.exp(17.269*AT2/(AT2+237.3))

#Actual Vapour Pressure
RH1=COPR_data['RH_avg_1']
RH1 = (RH1.ffill()+RH1.bfill())/2
RH1 = RH1.bfill().ffill()

RH2=COPR_data['RH_avg_2']
RH2 = (RH2.ffill()+RH2.bfill())/2
RH2 = RH2.bfill().ffill()


Ea1=(Es1 * RH1) / 100
Ea2=(Es2 * RH2) / 100

#Vapour Pressure Deficit 
VPD=Es2-Ea2

#%%Net Radiation 
Rnet=COPR_data['NetRs_Avg']+COPR_data['NetRl_Avg'] #ok

#%%Slope for vapour pressure curve
delta = 4098*(0.6108*np.exp((17.27*AT2)/(AT2+237.3)))/(AT2+237.3)**2 #ok

#%%Pressure
P=101.3*np.exp(-0.006/8.434) #ok

#Psychromatic Constant
pc=(1005*101.3)/(0.622*2468000) #ok

#%%Temperature in K at Height 1 and 2
TK1=AT1+273.16
TK2=AT2+273.16

#Density of air at height 2 (280cm)
R=287.05 #J/kgK #fixd for COPR

pa=P/(R*TK2)*1000 #ok

#%%Temperature at surface from longwave radiation
e=0.99 #Dar excel
sigma=0.0000000567 #Dar excel 
Lwup=COPR_data['CG3DnCo_Avg']
Tsurf=(Lwup/(e*sigma))**0.25 #ok

#Temperature at Soil Depth 1 in K
ST1=COPR_data['SoilTC_1_Avg']
ST1[ST1 == 0] = np.nan
ST1 = (ST1.ffill()+ST1.bfill())/2
ST1 = ST1.bfill().ffill()

Ts1= ST1+273.16 #ok

#Temperature difference between soil surface and soil depth 1
Tdiff=Ts1-Tsurf #ok

#%%Soil Heat Flux
K=0.4 #Dars excel
Dz=0.1 #Dars Excel 
G= K*Tdiff/Dz #ok

#%%delta ET
cp=1005 #(Jkg-1K-1) #fixed for COP from Dars excel 
ch= 0.21 #fixed for COPR - crop height 
zm=3.17 #fixed for COPR
d=(2/3)*ch #FAO formula
zom=0.123*ch #FAO formula 
zoh=0.1*ch #FAO formula 
zh=2.85 #Fixed for COPR
#%%
uz=COPR_data['WS_ms_Avg'] #Windspeed at height 2
uz = (uz.ffill()+uz.bfill())/2
uz = uz.bfill().ffill()
u2= uz*4.87/  np.log(67.8*3.17-5.42)

#%%
LAI=24*ch*0.5 #Leaf Area Index 

rl=200 #variable from Dars excel - stomatal resistance
ra=np.log((zm-d)/zom)*np.log((zh-d)/zoh) / 0.41**2 #aerodynamic resistance
rs=rl/LAI #bulk surface resistance
Ra=ra/uz
LV=2468000 #Dar
#%%Lambda ET / estimation of crop evapotranspiration = Etc
LET = (delta * (Rnet + G) + pa * cp * (VPD / Ra)) / (delta + pc * (1 + rs / Ra))

#%% Bowen Ratio
N = len(COPR_data['Date'])
B=np.zeros(N)

for t in range (N):
    if (Ea2[t] - Ea1[t] < 0):
        B[t] = (B[t-1]+ B[t+1])/2
    else:
        B[t] = P * cp *(AT2[t]-AT1[t]) / (LV * 0.622 * (Ea2[t]-Ea1[t]))
        
#%%AET through Bowen ratio
        
B = P * cp *(AT2-AT1) / (LV * 0.622 * (Ea2-Ea1))
LE = (Rnet + G)/(B+1)
RET_b = LE * 900 / LV
RET_b[RET_b < 0] = 0 
AET_b = RET_b.resample('D').sum()
#plt.plot(ET_b)

#%%Potential Crop Evapotranspiration with resistance parameters etc

PET=LET*86400/1000/LV*1000 #after Dars calculation (900 for 15 min, 9000 for 25 hours)

#%%CROP EVAPORATION 
PET_all=pd.DataFrame(PET)
PET_all.columns=['PET']
#PET_all=PET_all.set_index(pd.DatetimeIndex()) 
#PET_all['PET'][PET_all['PET'] < 0] = 0 
#PET_all = PET_all.resample('D').sum()
#PET_all=PET_all[["PET"]].astype(float).replace(0,np.nan)
#PET_all['Date'] = pd.to_datetime(PET_all.index)


#PET_all=PET_all.interpolate(method='time')

#%%
PET_all.to_csv("RET_COPR_CCS.csv",sep=';' ,date_format='%Y-%m-%d')

#%% Reference ET from grass surface (ET0/PET) daily timestep

Rnet= Rnet.resample('D').sum() * 0.0009
G = G.resample('D').sum() * 0.0009
VPD = VPD.resample('D').mean()
u2 = u2.resample('D').mean()
AT2=AT2.resample('D').max()
delta = (uz.resample('D').mean())*4.87 / np.log(67.8*3.17-5.42)

RET=(0.408*delta*(Rnet+G)+((pc*900*VPD*u2)/(AT2+273)))/(delta+pc*(1+0.34*u2))

#%% Dataset of PET and RAIN to be used for Bucket model - 15 min interval 

I_COPR=pd.concat([PET_all, COPR_Rain], axis=1)
I_COPR.to_csv("Infiltration_COPR180221.csv", sep=';') #updated 18.02.21

#%%




