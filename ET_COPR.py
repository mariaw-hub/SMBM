
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

# Convert UTC to PST
COPR_data=COPR_data.tz_localize('UTC')
COPR_data=COPR_data.tz_convert('US/Pacific')
#COPR_data = COPR_data[(COPR_data.index > '2007-06-01')]
COPR_data = COPR_data.loc['2007-12-01':'2019-09-30']

COPR_data[COPR_data == 'NAN'] = np.nan

for col in list(COPR_data.columns):
        COPR_data[col] = pd.to_numeric(COPR_data[col])
COPR_data['Date'] = pd.to_datetime(COPR_data['Date'])

COPR_Rain = COPR_data['Rain_mm_Tot'].resample('D').sum()
COPR_Rain = COPR_Rain.loc['2007-12-01':'2019-09-30']

#%%########################PENMAN MONTHEITH CALCULATION FOR PET##################

#Saturation Vapour Pressure 
AT1=COPR_data['AirTC_1_Avg'] + 4.0
AT2=COPR_data['AirTC_2_Avg'] +4.0


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

PET=LET*900/1000/LV*1000 #after Dars calculation 

#%%CROP EVAPORATION 
PET_all=pd.DataFrame(PET)
PET_all.columns=['PET']
PET_all=PET_all.set_index(pd.DatetimeIndex(COPR_data.index)) #creates the Index based on Date
PET_all=PET_all.loc['2007-12-01':'2019-09-30']
PET_all['PET'][PET_all['PET'] < 0] = 0 


#AET_all = AET_all.loc['2008-01-01':'2018-12-31']
PET_all = PET_all.resample('M').sum()
PET_all = PET_all.mask(PET_all['PET'].between(0,0.01))
PET_all['Date'] = pd.to_datetime(PET_all.index)
#PET_all_M = PET_all['PET'].resample('M').sum()
#%%
PET_all.to_csv("ET_COPR_CC.csv",sep='\t')

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
I_COPR.to_csv("Infiltration_COPR020620.csv", sep='\t')


#%%Cumulative P minus ET to show moisture deficit  - input data is in DAILY format 
#CWBI = sum(P-ET0)

I_COPR = pd.read_csv("Infiltration_COPR311019.csv", sep='\t')
I_COPR['Date'] = pd.to_datetime(I_COPR['Date'])
I_COPR=I_COPR.set_index(pd.DatetimeIndex(I_COPR['Date'])) 
I_COPR = I_COPR.resample('M').sum()
I_COPR = I_COPR.loc['2007-10-01' : '2019-09-30']
I_COPR['Year'] = I_COPR.index.year
I_COPR['NET_P'] = (I_COPR['Rain'] - I_COPR['AET']).groupby(I_COPR['Year']).cumsum()

I_AIRS = pd.read_csv("Infiltration_AIR011119.csv", sep='\t')
I_AIRS['Date'] = pd.to_datetime(I_AIRS['Date'])
I_AIRS=I_AIRS.set_index(pd.DatetimeIndex(I_AIRS['Date'])) 
I_AIRS = I_AIRS.resample('M').sum()
I_AIRS = I_AIRS.loc['2007-10-01' : '2019-09-30']
I_AIRS['Year'] = I_AIRS.index.year
I_AIRS['NET_P'] = (I_AIRS['Rain'] - I_AIRS['AET']).groupby(I_AIRS['Year']).cumsum()



#%%
fig = plt.figure()
plt.plot(I_COPR['NET_P'], color = 'blue', label = 'Coastal')
plt.plot(I_AIRS['NET_P'], color ='orange', label ='Inland')
plt.axhline(0, color = 'red', alpha=0.8)
plt.ylabel('P - PET (mm/yr)', fontsize='large', fontweight='bold')
plt.yticks(fontsize='medium')
plt.xticks(pd.date_range('2008','2019', freq='AS'), fontsize='medium')
plt.axvspan('2012-01-01', '2019-01-01', color = 'grey', alpha = 0.3)
fig.legend(fontsize='x-small', ncol= 3, loc ='lower center', frameon=False)
plt.show()








