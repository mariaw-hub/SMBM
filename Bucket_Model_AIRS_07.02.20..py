import numpy as np #Use this for number crunching
import pandas as pd 
import datetime as dt
import matplotlib.pyplot as plt #Use this for plotting
import matplotlib.dates as mdates #used for plotting
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.stats import uniform
from numpy import genfromtxt
from sklearn.preprocessing import normalize
######################## LOAD INPUT DATA  #####################################
#%%Load formatted xls file
data = pd.read_csv('Infiltration_AIR070420.csv', sep='\t')

data['Date'] = pd.to_datetime(data['Date'], utc=True)
data=data.set_index(pd.DatetimeIndex(data['Date'])) 
data = data.loc['2008-01-01':'2019-09-30']
data['Date'] = pd.to_datetime(data['Date'])

#Names of columns that we want
varnames = ['Date', 'PET','Rain_mm_Tot']
#Extract numpy arrays from variables in dataframe
series = [column.values for label, column in data[varnames].items()]
#Make separate names for each - make new np array for calc
Date, RET, Rain = series

N = len(Date) #Number of timesteps in simulation
#%% #kc from file starts in Jan 2007, but can be cut down, run model from Oct 2007
kc_new = genfromtxt('kc_both.csv', delimiter=',')
kc_new=pd.DataFrame(kc_new)
kc_new=kc_new.set_index(pd.DatetimeIndex(data['Date'])) 

#kc_new = np.asarray(kc_new.drop(data[data['PET'].isna()].index))

#data=data.dropna()
series = [column.values for label, column in data[varnames].items()]
#Make separate names for each - make new np array for calc
Date, RET, Rain = series
N = len(data.index) #Number of timesteps in simulation

#%%######################## SOIL MOISTURE SIMULATION ############################
#Allocate some arrays for results -size 1000 because of Monte Carlo
R = np.zeros(N+1)         #initiate retention array
cumR = np.zeros(N+1)      #initiate cumulative retention array
cumP = np.zeros(N+1)      #initiate cumulative retention array
D = np.zeros([N+1, 1000])  #Drainage array      
SMD = np.zeros([N+1,1000]) #SMD array
PET = np.zeros(N+1)        
AET = np.zeros([N+1,1000])       
Ks = np.zeros([N+1,1000])        
P = np.zeros(N)             
MB = np.zeros([N+1, 1000])        
TAW = np.zeros([N+1,1000])
RAW = np.zeros([N+1,1000])
NSE=np.empty(1000)
Theta=np.zeros([N+1, 1000])
#%% Input parameters - change as needed
SMD[0] =  100               #initial soil moisture deficit (mm)
#Pc = 0.5                  #Depletion Fraction, depends on the crop/ evaporation rate, , average fraction of
                            #TAW that can be extracted before moisture stress/reduction in ET
#Zr = 800                  #depth of rooted profile (mm) - depends on crop type- chosen Fodder for COPR
#Ze = 150                  #evaporation depth, can be between 0.1-0.25m
#B = 0                  # Bare soil proportion - coastal has less bare soil - currently set to 0
#%% Range of factors to run Monte Carlo variations - SOIL TYPE from Measurements
#mean value with a stdev to capture possible range above and below 
#fc = np.random.uniform(0.2,0.3, 1000)
#wp = np.random.uniform(0.001,0.05, 1000)
#Pc = np.random.uniform(0.3,0.7, 1000)
#Zr = np.random.uniform(700,1000,1000)
#%%
#np.savetxt("fc_airs.csv",fc, delimiter=',')
#np.savetxt("Pc_airs.csv",Pc, delimiter=',')
#np.savetxt("wp_airs.csv", wp, delimiter=',')
#np.savetxt("Zr_airs.csv",Zr, delimiter=',')
#%%Load the distributions
fc = genfromtxt('fc_airs.csv', delimiter=',')
Pc = genfromtxt('Pc_airs.csv', delimiter=',')
Zr = genfromtxt('Zr_airs.csv', delimiter=',')
wp= genfromtxt('wp_airs.csv', delimiter = ',')

#%% calculate other parameters
TAW =  (fc - wp) * Zr #*(1-B) + (fc_range - 0.5*wp_range)*Ze *B  #total available water (mm)
RAW = Pc * TAW               #readily available water (mm)
P = Rain 
PET = RET * kc_new[0]                
#%% THE MODEL 

for i in range (1000):
    for t in range(N):      

        Ks[t,i] = (TAW[i] - SMD[t,i]) / (TAW[i] - RAW[i])  #calculate soil stress factor - note this is only used when RAW<SMD<TAW when it ranges from 0 to 1

#% case for when PET can be met by excess rainfall and/or a decrease in the SMD
        if P[t] > PET[t]:
            AET[t,i] = PET[t]
            if P[t] - PET[t] > SMD[t,i]:
                D[t,i]= P[t] - PET[t] - SMD[t,i] #i.e. drainage occurs
                SMD[t+1,i] = 0                 #i.e. SMD reduced to zero because drainage occurs
            else:                            #i.e. P>PET but not enough excess to cause drainage
                SMD[t+1,i] = SMD[t,i] + PET[t] - P[t]
                D[t,i] = 0
        else:                               #i.e. rest of calcs are for P<PET and therefore zero drainage
            D[t,i] = 0
            if SMD[t,i] < RAW[i]:
            # if crop NOT stressed
                AET[t,i] = PET[t]
                SMD[t+1,i] = SMD[t,i] + AET[t,i] - P[t]
            elif SMD[t,i] < TAW[i]:
            # if crop IS stressed, i.e. RAW < SMD < TAW.
                AET[t,i] = P[t] + Ks[t,i] * (PET[t] - P[t]) #i.e. SMD between RAW and TAW
                SMD[t+1,i] = SMD[t,i] + AET[t,i]  - P[t]
            else:
            #if wilting, i.e. SMD >= TAW
                AET[t,i] = P[t]
                SMD[t+1,i] = SMD[t,i]
        
        Theta[t,i]  = fc[i] - SMD[t,i] / Zr[i]

    print(i, SMD[:,i].min(), SMD[:,i].max()) 
    
#%% Modelled SM into Df - run from 2007 but cut down here for NSE
SM_mod_airs=pd.DataFrame(Theta)
SM_mod_airs.drop(SM_mod_airs.tail(1).index,inplace=True)
SM_mod_airs = SM_mod_airs.set_index(data.Date)
SM_mod_airs_M = SM_mod_airs.resample('M').mean()

#%% Calibration and validation periods
a = np.asarray(Air_SM1_M['SM 1'].loc['2008-01-01':'2013-12-31'])
b = np.asarray(SM_mod_airs_M.loc['2008-01-01':'2013-12-31'])

bad = ~np.logical_or(np.isnan(a), np.isnan(b[:,0]))

cal_obs=np.compress(bad, a) 
cal_sim =np.asarray((SM_mod_airs_M.loc['2008-01-01':'2013-12-31']).dropna())
#%%

val_obs = Air_SM1_M['SM 1'].loc[(Air_SM1_M.index < '2016-08-31') | (Air_SM1_M.index > '2018-02-28')]
val_obs = np.asarray(val_obs.loc['2014-01-01':'2019-09-30'])
val_sim = SM_mod_airs_M.loc[(SM_mod_airs_M.index < '2016-08-31') | (SM_mod_airs_M.index > '2018-02-28')]
val_sim = np.asarray(val_sim.loc['2014-01-01':'2019-09-30'])

#%%Select the models in the calibration period, 1. Ks Test, 2. NSE
o = cal_obs
s = cal_sim
s=s.T

p_values=[]
for i in range(1000):
    p_values.append(stats.ks_2samp(o,s[i]).pvalue)
    
p_values=np.asarray(p_values)  

#NASH SUTCLIFFE EFFICIENCY for the mean modelled SM against the observed SM
nse_1 = np.sum((o-np.mean(o))**2)

NSE=np.empty(1000)
for i in range(1000):
    NSE[i]=1-np.sum((s[i,:]-o)**2) / nse_1 
    
criteria = np.column_stack((NSE,p_values,s))  
# Use Critera to pick the best models in the calibration period
#drop the models that have a p< <0.01 and low NSE <0.5
cal_models=pd.DataFrame(criteria)
cal_models=cal_models.drop(cal_models[cal_models[1] < 0.01].index) #reject all the models with a low p-value
cal_models=cal_models.drop(cal_models[cal_models[0] < 0.5].index) #reject all the models with a low NSE value
#These models have passed the criteria in the calibration period

#%%Validate the models from the calibration period
val_sim=val_sim.T
val_sim = pd.DataFrame(val_sim)

val_sim = np.asarray(val_sim[val_sim.index.isin(cal_models.index)]) #all the models that were valid for the calibration period

o = val_obs
s = val_sim

N=len(cal_models)
p_values_val=[]
for i in range(N):
    p_values_val.append(stats.ks_2samp(o,s[i]).pvalue)
    
p_values_val=np.asarray(p_values_val)  

#NASH SUTCLIFFE EFFICIENCY for the mean modelled SM against the observed SM
nse_1 = np.sum((o-np.mean(o))**2)
NSE_val=np.empty(N)
for i in range(N):
    NSE_val[i]=1-np.sum((s[i,:]-o)**2) / nse_1 

criteria_val = np.column_stack((NSE_val,p_values_val, s))
val_models=pd.DataFrame(criteria_val)
val_models=val_models.drop(val_models[val_models[1] < 0.01].index) #reject all the models with a low p-value
val_models=val_models.drop(val_models[val_models[0] < 0.5].index) #reject all the models with a low NSE value
#the models from the calibration period have also passed the criteria in the validation period 

#%%GLUE analysis of the models that are valid in both periods 

GLUE_models = cal_models[cal_models.index.isin(val_models.index)]
add = val_models[val_models.index.isin(GLUE_models.index)]
NSE_val=GLUE_models[[0,1]]
NSE_val2 = add[[0,1]]
GLUE_models = np.asarray(GLUE_models.drop(GLUE_models.columns[[0,1]], axis=1)) #all the models that were valid for the calibration period
add = np.asarray(add.drop(add.columns[[0,1]], axis=1)) #all the models that were valid for the calibration period

GLUE_models = np.concatenate((GLUE_models, add), axis=1)


#%%NSE and ks for all the valid models in both periods - AIR_SM1_M df must be from 2008-01-01
ox=np.concatenate((cal_obs,val_obs),axis=0)

nse_1 = np.sum((ox-np.mean(ox))**2)
sx=GLUE_models
NSE_glue=np.empty(len(sx))

for i in range(len(sx)):
    NSE_glue[i]=1-np.sum((sx[i,:]-ox)**2) / nse_1 

p_values_glue=[]
for i in range(len(sx)):
    p_values_glue.append(stats.ks_2samp(ox,sx[i]).pvalue)
    
p_values_glue=np.asarray(p_values_glue)  # array of NSE and p values for all valid models
#%%
combined_NSE = np.asarray(NSE_glue * 1 / sum(NSE_glue)) #Probablity 
#%% Find best GLUE Model 
np.argmax(p_values_glue)
p_values_glue.max()
 #then pick that model from GLUE array to plot 

#%%sort the Models in ascending order without sorting the NSE, RMSE and Probability and keep the index
qw95=[]
qw05=[]

for i in range(122):
    ind=np.argsort((GLUE_models.T)[i,:])
    y=GLUE_models[ind,i]
    x=np.add.accumulate(combined_NSE[ind])
    x=(x-np.min(x))/(np.max(x)-np.min(x))
    f= interp1d(x, y, kind='linear')
    qw05.append(f(0.05))
    qw95.append(f(0.95))

#%%#%%Plotting uncertainty bands around the observed soil moisture
xaxis= SM_mod_airs_M.loc[(SM_mod_airs_M.index < '2016-08-31') | (SM_mod_airs_M.index > '2018-02-28')]
xaxis=xaxis.dropna()
xaxis=xaxis.index

SM_air= pd.DataFrame(ox)
SM_air = SM_air.set_index(pd.DatetimeIndex(xaxis)) 
best_model = (GLUE_models[[0],:]).T
best_model= (pd.DataFrame(best_model)).set_index(pd.DatetimeIndex(xaxis))

qw05 = (pd.DataFrame(qw05)).set_index(pd.DatetimeIndex((xaxis)))

qw95 = (pd.DataFrame(qw95)).set_index(pd.DatetimeIndex((xaxis)))


#%%
fig=plt.figure()
plt.plot(SM_air, color='orange', label='Observed')
plt.plot(best_model, color='black', label='Best Fit' )
p#lt.plot(pd.DataFrame(qw05), color='red',alpha=0.3)
#plt.plot(pd.DataFrame(qw95), color='red',alpha=0.3)
#plt.fill_between(xaxis, qw05,qw95, color='grey', alpha=0.5, label='Confidence Interval')   
plt.xticks(pd.date_range('2009','2020', freq='AS'), fontsize='medium')
plt.yticks(fontsize='medium')
plt.axhline(0.144, color = 'brown', alpha= 0.4)
plt.ylabel('VMC (m $\mathregular{^3}$/ m $\mathregular{^3}$)', fontsize='medium', fontweight='bold')
fig.legend(fontsize='x-small', ncol= 3, loc ='lower center', frameon=False)

#%%KDE plot for the best fit model to compare to hist. KDE
sns.kdeplot(ox, color='black')
plt.ylabel('KDE', fontweight='bold', fontsize='large')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.xlabel('VMC (m $\mathregular{^3}$/ m $\mathregular{^3}$)',fontweight='bold', fontsize='large' )
sns.kdeplot(GLUE_models[57].T, color='green')

#%%

fc_glue = pd.DataFrame(fc)
fc_glue = fc_glue[fc_glue.index.isin(cal_models.index)] #all the models that were valid for the calibration period
fc_glue = fc_glue[fc_glue.index.isin(val_models.index)] #all the models that were valid for the calibration period
fc_glue = np.asarray(fc_glue)
fc_glue.sort(axis=0)
wp_glue = pd.DataFrame(wp)
wp_glue = wp_glue[wp_glue.index.isin(cal_models.index)] #all the models that were valid for the calibration period
wp_glue = wp_glue[wp_glue.index.isin(val_models.index)] #all the models that were valid for the calibration period
wp_glue = np.asarray(wp_glue)
wp_glue.sort(axis=0)

Zr_glue = pd.DataFrame(Zr)
Zr_glue = Zr_glue[Zr_glue.index.isin(cal_models.index)] #all the models that were valid for the calibration period
Zr_glue = Zr_glue[Zr_glue.index.isin(val_models.index)] #all the models that were valid for the calibration period
Zr_glue = np.asarray(Zr_glue)
Zr_glue.sort(axis=0)

Pc_glue = pd.DataFrame(Pc)
Pc_glue = Pc_glue[Pc_glue.index.isin(cal_models.index)] #all the models that were valid for the calibration period
Pc_glue = Pc_glue[Pc_glue.index.isin(val_models.index)] #all the models that were valid for the calibration period
Pc_glue = np.asarray(Pc_glue)
Pc_glue.sort(axis=0)

#%%

