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
from sklearn.linear_model import LinearRegression
######################## LOAD INPUT DATA  #####################################
#%%Load formatted xls file
data = pd.read_csv('Infiltration_COPR070420.csv', sep=',')
data['Date'] = pd.to_datetime(data['Date'], utc=True)
data=data.set_index(pd.DatetimeIndex(data['Date'])) 
data = data.loc['2008-01-01':'2019-09-30']
data['Date'] = pd.to_datetime(data['Date'])
#Names of columns that we want
varnames = ['Date', 'PET','Rain_mm_Tot']
#Extract numpy arrays from variables in dataframe
     

 #%%Read kc_copr file - no need to to run the whole NDVI script again. 
#kc starts from 2008-01 until 2019-09 
 
kc_new = genfromtxt('kc_both.csv', delimiter=',')
kc_new=pd.DataFrame(kc_new)
kc_new=kc_new.set_index(pd.DatetimeIndex(data['Date'])) 

kc_new = np.asarray(kc_new.drop(data[data['PET'].isna()].index))

data=data.dropna()
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
SMD[0] = 40               #initial soil moisture deficit (mm)
#calculated from measured soil moisture content at first date 2008-01-01
#%% Range of factors to run Monte Carlo variations - SOIL TYPE from Measurements
#mean value with a stdev to capture possible range above and below 
#fc = np.random.uniform(0.4, 0.50, 1000)
#wp = np.random.uniform(0.1,0.2, 1000)
#Pc = np.random.uniform(0.3,0.7, 1000)
#Zr = np.random.uniform(700,1000,1000)
#%%Save MC variables to file and use them again every time 
#np.savetxt("fc_copr.csv",fc, delimiter=',')
#np.savetxt("wp_copr.csv",wp, delimiter=',')
#np.savetxt("Pc_copr.csv",Pc, delimiter=',')
#np.savetxt("Zr_copr.csv",Zr, delimiter=',')
#%%Load MC variables from data

fc = genfromtxt('fc_copr.csv', delimiter=',')
Pc = genfromtxt('Pc_copr.csv', delimiter=',')
Zr = genfromtxt('Zr_copr.csv', delimiter=',')
wp = genfromtxt('wp_copr.csv', delimiter=',')

#%% calculate other parameters
TAW =  (fc - wp) * Zr #* (1-B) + (fc - 0.5*wp)*Ze *B  #total available water (mm)
RAW = Pc * TAW               #readily available water (mm)
P = Rain   
PET = RET * kc_new[0]                
#%%MODEL RUNS FROM OCTOBER 2007 - Sept 2019 as of 01.11.2019 

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
        
        Theta[t,i] = fc[i] - SMD[t,i] / Zr[i] #to calculate SM from SMD
   
    print(i, SMD[:,i].min(), SMD[:,i].max()) 
    
#%% Modelled SM into Df - the model ran from october 2007 but cut down here for NSE to start in 2008
SM_mod=pd.DataFrame(Theta)
SM_mod.drop(SM_mod.tail(1).index,inplace=True)
SM_mod = SM_mod.set_index(data.Date)
SM_mod_M = SM_mod.resample('M').mean()
#SM_mod_M = SM_mod_M.dropna()
#%%Save output to file

#%% Modelled SMD into a dataframe - original version is from Oct 2007
cal_obs = np.asarray(COPR_SM1_M['SM 1'].loc['2008-02-01':'2013-12-31'])
cal_sim= np.asarray(SM_mod_M.loc['2008-02-01':'2013-12-31'])

#bad = ~np.logical_or(np.isnan(a), np.isnan(b[:,0]))

#cal_obs=np.compress(bad, a)  

#%%
bad = ~np.logical_or(np.isnan(np.asarray(COPR_SM1_M['SM 1'].loc['2014-01-01':'2019-09-30'])), 
                     np.isnan(np.asarray(SM_mod_M.loc['2014-01-01':'2019-09-30'])[:,0]))

val_obs=np.compress(bad, np.asarray(COPR_SM1_M['SM 1'].loc['2014-01-01':'2019-09-30'])) 
val_sim =np.asarray(SM_mod_M.loc['2014-01-01':'2019-09-30'].dropna())


#%% Kolmogorov Smirnoff test  to select models that follow same distribution as observed SM
o = cal_obs
s = cal_sim
s=s.T

p_values=[]
for i in range(1000):
    p_values.append(stats.ks_2samp(o,s[i]).pvalue)
    
p_values=np.asarray(p_values)  #list of all the p-values of the 5000 runs

#NASH SUTCLIFFE EFFICIENCY for the mean modelled SM against the observed SM
nse_1 = np.sum((o-np.mean(o))**2)

NSE=np.empty(1000)
for i in range(1000):
    NSE[i]=1-np.sum((s[i,:]-o)**2) / nse_1 
    
criteria = np.column_stack((NSE,p_values,s))  #criter of rejection during calibration period

# Use Critera to pick the best models in the calibration period
#drop the models that have a p< <0.05 and low NSE
cal_models=pd.DataFrame(criteria)
cal_models=cal_models.drop(cal_models[cal_models[1] < 0.01].index) #reject all the models with a low p-value
cal_models=cal_models.drop(cal_models[cal_models[0] < 0.5].index) #reject all the models with a low NSE value

#cal_models = all the models that passed selection in the calibration period

#%%Validate the models from the calibration period

val_sim=val_sim.T
val_sim = pd.DataFrame(val_sim)

val_sim = np.asarray(val_sim[val_sim.index.isin(cal_models.index)]) 
#select all the models that were valid for the calibration period for validation 

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

#these models have passed validation 
#%%GLUE analysis of the models that are valid in both periods 

GLUE_models_c = cal_models[cal_models.index.isin(val_models.index)] 
#combine and drop models that are valid in both periods
add = val_models[val_models.index.isin(GLUE_models_c.index)]

NSE_val=GLUE_models_c[[0,1]]
NSE_val2 = add[[0,1]]
GLUE_models_c = np.asarray(GLUE_models_c.drop(GLUE_models_c.columns[[0,1]], axis=1)) 
#all the models that were valid for the calibration period
add = np.asarray(add.drop(add.columns[[0,1]], axis=1))
 #all the models that were valid for the calibration period

GLUE_models_c = np.concatenate((GLUE_models_c, add), axis=1) 
#these are all valid models
#%% NSE and Ks for all the valid models in both periods - to find best fit
ox=np.concatenate((cal_obs,val_obs),axis=0)
sx=GLUE_models_c

nse_1 = np.sum((ox-np.mean(ox))**2)
NSE_glue=np.empty(len(sx)) #ignore error message

for i in range(len(sx)):
    NSE_glue[i]=1-np.sum((sx[i,:]-ox)**2) / nse_1 

p_values_glue=[]
for i in range(len(sx)):
    p_values_glue.append(stats.ks_2samp(ox,sx[i]).pvalue)
    
p_values_glue=np.asarray(p_values_glue)  
   

#%%Combine parameter uncertainy for the cal & val models separately - or use the 1 NSE for the combined df?

combined_NSE = np.asarray(NSE_glue * 1 / sum(NSE_glue))
#%%sort the Models in ascending order without sorting the NSE, RMSE and Probability and keep the index
qw95_c=[]
qw05_c=[]

for i in range(139): #139 is lenght of time series - stays fixed regardless of how many models are valid
    ind=np.argsort((GLUE_models_c.T)[i,:])
    y=GLUE_models_c[ind,i]
    x=np.add.accumulate(combined_NSE[ind])
    x=(x-np.min(x))/(np.max(x)-np.min(x))
    f= interp1d(x, y, kind='linear')
    qw05_c.append(f(0.05))
    qw95_c.append(f(0.95))

#%% Find best GLUE Model 
np.argmax(p_values_glue)
p_values_glue.max()
 #then pick that model from GLUE array to plot 
#%%Plotting uncertainty bands around the observed soil moisture
SM_c= np.concatenate((cal_obs,val_obs),axis=0)
xaxis_c=(SM_mod_M.loc['2008-02-01':].dropna()).index
SM_c= (pd.DataFrame(SM_c)).set_index(pd.DatetimeIndex((xaxis_c)))

best_model_c = (GLUE_models_c[7]).T #chose on base of p_value, is from Feb 2008 - Oct 2019, with no Oct 2017
best_model_c= (pd.DataFrame(best_model_c)).set_index(pd.DatetimeIndex((xaxis_c)))

#qw05_c = (pd.DataFrame(qw05_c)).set_index(pd.DatetimeIndex((xaxis_c)))

#qw95_c = (pd.DataFrame(qw95_c)).set_index(pd.DatetimeIndex((xaxis_c)))

SM_mod_M_std = SM_mod_M.std(axis=1)
SM_mod_M_std=SM_mod_M_std[SM_mod_M_std.index.isin(best_model_c.index)] 

SM_std_air=SM_mod_airs_M.std(axis=1)
SM_std_air=SM_std_air[SM_std_air.index.isin(pd.DataFrame(GLUE_models_c[7]).index)] 

#%% plots AIRS and COPR models together 

fig=plt.figure()
plt.subplot(211)
plt.plot(best_model_c[0], color='black', label='Best Fit', linestyle='--' )
plt.fill_between(SM_mod_M_std.index,best_model_c[0] + SM_mod_M_std, best_model_c[0] - SM_mod_M_std, color = 'grey', alpha = 0.6)
plt.axhline(0.13, color='green')
plt.plot(SM_c, color='blue', label='Observed')
plt.yticks(fontsize='medium')
plt.ylabel('VMC (m $\mathregular{^3}$/ m $\mathregular{^3}$)', fontsize='medium', fontweight='bold')
plt.ylim(0.05,0.5)

plt.subplot(212)
plt.plot(SM_air, color='orange', label='_nolegend_')
plt.plot(SM_mod_airs_M[29], color='black', label='Best Fit',linestyle='--' )
plt.axhline(0.07, color = 'green')
plt.fill_between(SM_std_air.index,SM_mod_airs_M[29] + SM_std_air, SM_mod_airs_M[29] - SM_std_air, color = 'grey', alpha = 0.6)
plt.yticks(fontsize='medium')
plt.ylabel('VMC (m $\mathregular{^3}$/ m $\mathregular{^3}$)', fontsize='medium', fontweight='bold')
plt.ylim(0,0.3)

fig.legend(fontsize='x-small', ncol= 4, loc ='lower center', frameon=False)




#%%
sns.kdeplot(GLUE_models_c[0].T, alpha = 0.4)
sns.kdeplot(GLUE_models_c[1].T,alpha = 0.4)
sns.kdeplot(GLUE_models_c[2].T,alpha = 0.4)
sns.kdeplot(GLUE_models_c[3].T,alpha = 0.4)
sns.kdeplot(GLUE_models_c[4].T,alpha = 0.4)
sns.kdeplot(GLUE_models_c[5].T,alpha = 0.4)
sns.kdeplot(GLUE_models_c[6].T,alpha = 0.4)
sns.kdeplot(GLUE_models_c[7].T,alpha = 0.4)
sns.kdeplot(np.asarray(SM_c[0]), color='blue')
sns.kdeplot(best_model_c[0], color='black')

plt.yticks(fontsize = 'medium')
plt.ylabel('KDE', fontweight='bold', fontsize='large')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.xlabel('VMC (m $\mathregular{^3}$/ m $\mathregular{^3}$)',fontweight='bold', fontsize='large' )
sns.kdeplot(np.asarray(SM_air[0]), color='orange')
plt.ylabel('KDE', fontweight='bold', fontsize='large')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.xlabel('VMC (m $\mathregular{^3}$/ m $\mathregular{^3}$)',fontweight='bold', fontsize='large' )
sns.kdeplot(best_model[0], color='black')

#sns.kdeplot(ox,GLUE_models[161].T)
#%%include uncertainty of 10% for equipment variability? 

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
#%% Count the times SM is below browning threshold





