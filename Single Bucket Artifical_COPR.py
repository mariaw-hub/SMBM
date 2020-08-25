import numpy as np #Use this for number crunching
import pandas as pd #Use this for IO and data prep
import matplotlib.pyplot as plt #Use this for plotting
import matplotlib.dates as mdates #used for plotting
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.stats import uniform
from numpy import genfromtxt
from datetime import datetime

######################## LOAD INPUT DATA  #####################################
#%%Load formatted xls file
data_art = pd.read_csv('Climate_scenario_COPR.csv', sep=',')
data_art['Date'] = pd.to_datetime(data_art['Date'], utc=True)

data_art=data_art.set_index(pd.DatetimeIndex(data_art['Date'])) 
data_art = data_art.resample('M').sum()

PET = pd.read_csv('Infiltration_COPR070420.csv', sep=',')
PET['Date'] = pd.to_datetime(PET['Date'], utc=True)
PET=PET.set_index(pd.DatetimeIndex(PET['Date'])) 
PET = PET.loc['2011-10-01':'2018-12-31']
PET = PET.resample('M').sum()
PET = PET.drop(['Rain_mm_Tot'], axis=1) #FILL IN GAP IN 2017 with 61.8904
PET.loc['2017-10-31'] = 61.8904

data_art = pd.concat([data_art, PET['PET']], axis=1) #df with original PET
data_art['P3'] = pd.DataFrame(data_art['Rain'] *0.9)

 #dry_months = PET.assign(x=PET.index.strftime('%m-%d')) \
               #.query("'03-01' <= x <= '10-31'").drop('x',1)
#dry_months = pd.DataFrame(dry_months['PET'] *1.10) #increase the dry season PET by 25 %

#PET_art= dry_months.combine_first(PET) #PET for Scneario C
#%%
PET_art = pd.read_csv('Infiltration_COPR070420.csv', sep=',') #PET with 10% change for CC3
PET_art['Date'] = pd.to_datetime(PET_art['Date'], utc=True)
PET_art=PET_art.set_index(pd.DatetimeIndex(PET_art['Date'])) 
PET_art = PET_art.loc['2011-10-01':'2018-12-31']
PET_art = PET_art.resample('M').sum()
PET_art = PET_art.drop(['Rain_mm_Tot'], axis=1)
PET_art.loc['2017-10-31'] = 61.8904
PET_art = PET_art * 1.1
#FILL IN GAP IN 2017 with 61.8904
#for simulation data_Art needs to start in January too - just reset index to data_art index

#%%
art_ndvi_copr = pd.DataFrame()
 #ndvi values give values from January 
art_ndvi_copr['NetP1'] = data_art['P1'] - data_art['PET']
art_ndvi_copr['3M1'] = art_ndvi_copr.iloc[:,0].rolling(window=3).sum() #december = january ndvi

art_ndvi_copr['NetP2'] = data_art['P2'] - data_art['PET']
art_ndvi_copr['3M2'] = art_ndvi_copr.iloc[:,2].rolling(window=3).sum()

art_ndvi_copr['NetP3'] = data_art['P3'] - PET_art['PET']
art_ndvi_copr['3M3'] = art_ndvi_copr.iloc[:,4].rolling(window=3).sum()


art_ndvi_copr['ndvi1'] = 0.45583 + 0.00056*art_ndvi_copr['3M1'] #from NDVI & Net P regression 
art_ndvi_copr['ndvi2'] = 0.45583 + 0.00056*art_ndvi_copr['3M2']
art_ndvi_copr['ndvi3'] = 0.45583 + 0.00056*art_ndvi_copr['3M3']


cols = ['ndvi1','ndvi2', 'ndvi3']
art_ndvi_copr[art_ndvi_copr[cols] < 0] = 0        

art_ndvi_copr['Kc1']= 2.03*art_ndvi_copr['ndvi1'] - 0.15 #from NDVI & Net P regression
art_ndvi_copr['Kc2']= 2.03*art_ndvi_copr['ndvi2'] - 0.15 
art_ndvi_copr['Kc3']= 2.03*art_ndvi_copr['ndvi3'] - 0.15 

art_ndvi_copr = art_ndvi_copr.loc['2011-12-01':'2018-11-30'] #equals january to december

#%%
data_art = data_art.loc['2012-01-01':'2018-12-31']
#Names of columns that we want
varnames = ['Rain','P1','P2','PET', 'P3']
#Extract numpy arrays from variables in dataframe
series = [column.values for label, column in data_art[varnames].items()]
#Make separate names for each - make new np array for calc
Rain, P1, P2, RET, P3= series


N = len(data_art.index) #Number of timesteps in simulation

art_ndvi_copr = art_ndvi_copr.set_index(data_art.index)
PET_art = PET_art.loc['2012-01-01':'2018-12-31'] #FILL IN THAT GAP IN 2017-OCT with 71.0773


#%%######################## SOIL MOISTURE SIMULATION ############################
#Allocate some arrays for results -size 1000 because of Monte Carlo
R = np.zeros(N+1)         #initiate retention array
cumR = np.zeros(N+1)      #initiate cumulative retention array
cumP = np.zeros(N+1)      #initiate cumulative retention array
D = np.zeros(N+1)  #Drainage array      
SMD = np.zeros(N+1) #SMD array
PET = np.zeros(N+1)        
AET = np.zeros(N+1)       
Ks = np.zeros(N+1)        
P = np.zeros(N)             
MB = np.zeros(N+1)        
TAW = np.zeros(N+1)
RAW = np.zeros(N+1)
NSE=np.empty(1000)
Theta=np.zeros(N+1)
#%% Input parameters - change as needed
SMD[0] =  100             #initial soil moisture deficit (mm)
#%%Parameters from the best model 
fc=0.403
wp=0.10
Pc=0.22
Zr=705

#%% calculate other parameters
TAW =  (fc - wp) * Zr #total available water (mm)
RAW = Pc * TAW               #readily available water (mm)
#P = P1
P = Rain #P varies with each scenario 1,2,3
#PET = RET * art_ndvi_copr['Kc1']
               #PET=AET in the model #artifical kc varies with each scenario 1,2,3
PET = PET_art['PET'] *art_ndvi_copr['Kc3']     #for artifical PET scenario         
#%%MODEL RUNS FROM OCTOBER 2007 - Sept 2019 as of 01.11.2019 

for t in range(N):      

        Ks[t] = (TAW - SMD[t]) / (TAW- RAW)  #calculate soil stress factor - note this is only used when RAW<SMD<TAW when it ranges from 0 to 1

#% case for when PET can be met by excess rainfall and/or a decrease in the SMD
        if P[t] > PET[t]:
            AET[t] = PET[t]
            if P[t] - PET[t] > SMD[t]:
                D[t]= P[t] - PET[t] - SMD[t] #i.e. drainage occurs
                SMD[t+1] = 0                 #i.e. SMD reduced to zero because drainage occurs
            else:                            #i.e. P>PET but not enough excess to cause drainage
                SMD[t+1] = SMD[t] + PET[t] - P[t]
                D[t] = 0
        else:                               #i.e. rest of calcs are for P<PET and therefore zero drainage
            D[t] = 0
            if SMD[t] < RAW:
            # if crop NOT stressed
                AET[t] = PET[t]
                SMD[t+1] = SMD[t] + AET[t] - P[t]
            elif SMD[t] < TAW:
            # if crop IS stressed, i.e. RAW < SMD < TAW.
                AET[t] = P[t] + Ks[t] * (PET[t] - P[t]) #i.e. SMD between RAW and TAW
                SMD[t+1] = SMD[t] + AET[t]  - P[t]
            else:
            #if wilting, i.e. SMD >= TAW
                AET[t] = P[t]
                SMD[t+1] = SMD[t]
        
        Theta[t] = fc- SMD[t] / Zr #to calculate SM from SMD
   
    
#%% Modelled SM into Df - the model ran from october 2007 but cut down here for NSE to start in 2008
SM_mod=pd.DataFrame(Theta)

SM_mod.drop(SM_mod.tail(1).index,inplace=True)
SM_mod = SM_mod.set_index(data_art.index)
SM_art3= SM_mod

#%% Modelled SMD into a dataframe - 
#SMD_s=pd.DataFrame(SMD)

#SMD_s.drop(SMD_s.tail(1).index,inplace=True)
#SMD_s = SMD_s.set_index(data_art.Date)
#SMD_s = SMD_s.loc['2014-01-01':'2016-12-31'] 

#SMD_M = SMD_s.resample('M').mean()
#SMD_M = SMD_M.mean(axis=1)

#%% create a mean SMD and mean SM of all the MCs to plot over 2 years

t=data_art.index #starts from 2012 January (becasue NDVI is from january too and as a result SM)
diff1 = data_art.P2 - data_art.P1
diff2 = data_art.P1 - data_art.Rain
diff3= data_art.P3 -data_art.Rain
months = mdates.MonthLocator()
y=best_model_c[0].loc['2012-01-01':'2018-12-31']
#%% Plot standard deviation of all 1000 Models around the best model plot as + - 1 STD

SM_mod_M_std = SM_mod_M.std(axis=1)
SM_mod_M_std=SM_mod_M_std[SM_mod_M_std.index.isin(best_model_c.index)] 
SM_mod_M_std = SM_mod_M_std.loc['2012':'2018']
#%%
fig = plt.figure() #original SM series with original P
ax1=fig.add_subplot(411)

ax1.plot(y, color='blue')   
ax1.fill_between(SM_mod_M_std.index,y + SM_mod_M_std, y - SM_mod_M_std, color = 'grey', alpha = 0.6)
ax1.set_ylabel('VMC (m $\mathregular{^3}$/ m $\mathregular{^3}$)', fontsize = 'small', fontweight='bold')
ax1.tick_params(axis= 'both', which ='major', labelsize='8')
ax1.yaxis.set_ticks(np.arange(0,0.6,0.1))
ax1.set_ylim(0.05,0.5)
ax1.xaxis.set_minor_locator(months)
ax1.axhline(0.17, color='black', alpha = 0.6)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.invert_yaxis()
ax2.bar(t, data_art.Rain, color='black', width = 20, alpha =0.6, label = 'Precipitation')
ax2.set_ylabel('P (mm/month)', fontsize = 'small',fontweight='bold')  # we already handled the x-label with ax1
ax2.tick_params(axis='y', which = 'major', labelsize ='8')
ax2.set_ylim(250,0)


ax3 = fig.add_subplot(412) #artifical series with scenario P1
ax3.set_ylabel('VMC (m $\mathregular{^3}$/ m $\mathregular{^3}$)', fontsize = 'small',fontweight='bold')
ax3.plot(SM_art1[0], color = 'blue')
ax3.tick_params(axis= 'both', which ='major', labelsize='8')
ax3.yaxis.set_ticks(np.arange(0,0.6,0.1))
ax3.set_ylim(0.05,0.5)
ax3.xaxis.set_minor_locator(months)
ax3.axhline(0.17, color='black', alpha = 0.6)

ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
ax4.invert_yaxis()
ax4.bar(t, data_art.P1, color='black', width = 20, alpha = 0.6, label = 'Precipitation')
ax4.bar(t, diff2, color='red', bottom = data_art.Rain, width = 20,label = 'Precipitation')
ax4.set_ylabel('P (mm/month)', fontsize = 'small',fontweight='bold')  # we already handled the x-label with ax1
ax4.tick_params(axis='y', which = 'major', labelsize ='8')
ax4.set_ylim(250,0)


ax7 = fig.add_subplot(413) #artifical series with scenario P2
ax7.set_ylabel('VMC (m $\mathregular{^3}$/ m $\mathregular{^3}$)', fontsize = 'small',fontweight='bold')
ax7.plot(SM_art2[0], color = 'blue')
ax7.tick_params(axis= 'both', which ='major', labelsize='8')
ax7.set_ylim(0.05,0.5)
ax7.xaxis.set_minor_locator(months)
ax7.yaxis.set_ticks(np.arange(0,0.6,0.1))
ax7.axhline(0.17,color='black', alpha = 0.6)

ax8 = ax7.twinx()  # instantiate a second axes that shares the same x-axis
ax8.invert_yaxis()
ax8.bar(t, data_art.P2, color='black', width = 20, alpha = 0.6, label = 'Precipitation')
ax8.bar(t, diff1, color='green', bottom = data_art.P1, width = 20,label = 'Precipitation')
ax8.bar(t, diff2, color='red', bottom = data_art.Rain, width = 20,label = 'Precipitation')
ax8.set_ylabel('P (mm/month)', fontsize = 'small',fontweight='bold')  # we already handled the x-label with ax1
ax8.tick_params(axis='y', which = 'major', labelsize ='8')
ax8.set_ylim(250,0)

ax9 = fig.add_subplot(414) #artifical series with scenario P2
ax9.set_ylabel('VMC (m $\mathregular{^3}$/ m $\mathregular{^3}$)', fontsize = 'small',fontweight='bold')
ax9.plot(SM_art3[0], color = 'blue')
ax9.tick_params(axis= 'both', which ='major', labelsize='8')
ax9.set_ylim(0.05,0.5)
ax9.set_xlabel('Date', fontsize = 'small', fontweight='bold')
ax9.xaxis.set_minor_locator(months)
ax9.yaxis.set_ticks(np.arange(0,0.6,0.1))
ax9.axhline(0.17, color='black', alpha = 0.6)

ax10 = ax9.twinx()  # instantiate a second axes that shares the same x-axis
ax10.invert_yaxis()
ax10.bar(t, data_art.Rain, color='black', width = 20, alpha = 0.6, label = 'Precipitation')
#ax10.bar(t, diff3, color='red', bottom = data_art.Rain, width = 20,label = 'Precipitation')
ax10.set_ylabel('P (mm/month)', fontsize = 'small',fontweight='bold')  # we already handled the x-label with ax1
ax10.tick_params(axis='y', which = 'major', labelsize ='8')
ax10.set_ylim(250,0)

plt.show()
#%% Distribution plots of synthetic NDVI

sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
plt.subplot(211)
sns.kdeplot(art_ndvi_copr['ndvi1'])
sns.kdeplot(art_ndvi_copr['ndvi2'], linestyle='--')
sns.kdeplot(art_ndvi_copr['ndvi3'])
plt.yticks(fontsize = 'medium')
plt.ylabel('KDE', fontweight='bold', fontsize='large')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.subplot(212)
sns.kdeplot(art_ndvi_airs['ndvi1'])
sns.kdeplot(art_ndvi_airs['ndvi2'],linestyle='--')
sns.kdeplot(art_ndvi_airs['ndvi3'])
plt.yticks(fontsize = 'medium')
plt.ylabel('KDE', fontweight='bold', fontsize='large')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.xlabel('NDVI', fontweight='bold', fontsize='large')


#%%
best_model_c['Scenario'] = 'SH'
bmc=pd.DataFrame(best_model_c.loc['2012':'2018'])

SM_art1['Scenario'] = 'A'
SM_art2['Scenario'] = 'B'
SM_art3['Scenario'] = 'C'

Art_C =pd.concat([bmc, SM_art1,SM_art2,SM_art3])

ax1=sns.violinplot(x='Scenario', y=0, data=Art_C, linewidth = 0.6, hue='Scenario', dodge=False) 
ax1.axhline(0.17, color='black',alpha=0.4)
#%%

stats.ks_2samp(best_model_c[0].loc['2012':'2018'], SM_art3[0])



