import numpy as np #Use this for number crunching
import pandas as pd #Use this for IO and data prep
import matplotlib.pyplot as plt #Use this for plotting
import matplotlib.dates as mdates #used for plotting
from scipy import stats
import seaborn as sns
from numpy import genfromtxt
from datetime import datetime

######################## LOAD INPUT DATA  #####################################
#%%Load formatted xls file
data_art = pd.read_csv('Climate_scenario_copr.csv',sep=',')
data_art['Date'] = pd.to_datetime(data_art['Date'], utc=True)

data_art=data_art.set_index(pd.DatetimeIndex(data_art['Date'])) 
#data_art = data_art.resample('M').sum()

PET = pd.read_csv('RET_COPR_03032021.csv', sep=';')
PET['Date'] = pd.to_datetime(PET['Date'], utc=True)
PET=PET.set_index(pd.DatetimeIndex(PET['Date'])) 
PET = PET.loc['2011-10-01':'2018-12-31']
#PET = PET.resample('M').sum()
#PET = PET.drop(['Rain_mm_Tot'], axis=1) #FILL IN GAP IN 2017 with 61.8904

data_art = pd.concat([data_art, PET['PET']], axis=1) #df with original PET

data_art['P3'] = data_art['P1'] 
data_art['P3'] = data_art['P3'] - data_art['P3']*0.25 #reduce monthly P by 25%

 #dry_months = PET.assign(x=PET.index.strftime('%m-%d')) \
               #.query("'03-01' <= x <= '10-31'").drop('x',1)
#dry_months = pd.DataFrame(dry_months['PET'] *1.10) #increase the dry season PET by 25 %

#PET_art= dry_months.combine_first(PET) #PET for Scneario C
        
#%%
PET_art = pd.read_csv('RET_COPR_CCS.csv',sep=';') #PET with 10% change for CC3
PET_art['Date'] = pd.to_datetime(PET_art['Date'], utc=True)
PET_art=PET_art.set_index(pd.DatetimeIndex(PET_art['Date'])) 
#PET_art = PET_art.loc['2011-12-01':'2018-21-31']
#PET_art = PET_art.resample('M').sum()
PET_art=PET_art.set_index(pd.DatetimeIndex(data_art['Date'])) 
PET_art.columns=['Date','PET_art']
data_art = pd.concat([data_art, PET_art['PET_art']], axis=1) #df with original PET

#%% Create Artifical NDVI from NETp as leading indicator, based on NDVI/NEtP regression
#Then use Artifcal NDVI to drive kc using the original kc equation

art_aP_copr = pd.DataFrame()
 #ndvi values give values from January 
art_aP_copr['aP1'] = data_art['P1'] - data_art['PET'] #Scenario A
art_aP_copr['3M1'] = art_aP_copr.iloc[:,0].rolling(window=3).sum() #december = january ndvi

art_aP_copr['aP2'] = data_art['P2'] - data_art['PET']
art_aP_copr['3M2'] = art_aP_copr.iloc[:,2].rolling(window=3).sum() #Scenario B

art_aP_copr['aP3'] = data_art['P3'] - data_art['PET_art']
art_aP_copr['3M3'] = art_aP_copr.iloc[:,4].rolling(window=3).sum() #scenario C

art_aP_copr['aP4'] = data_art['Rain'] - data_art['PET']
art_aP_copr['3M4'] = art_aP_copr.iloc[:,6].rolling(window=3).sum() #historic 

art_aP_copr=art_aP_copr.loc['2011-12-01':'2018-12-31'] #dec-feb ap = jan-jan ndvi

#%%
art_ndvi_copr=pd.DataFrame()

art_ndvi_copr['ndvi1'] = 0.006*art_aP_copr['3M1'] +0.4796 #from NDVI & aP regression 
art_ndvi_copr['ndvi2'] = 0.006*art_aP_copr['3M2'] +0.4796 #from NDVI & aP regression 
art_ndvi_copr['ndvi3'] = 0.006*art_aP_copr['3M3'] +0.4796 #from NDVI & aP regression 
art_ndvi_copr['ndvi4'] = 0.006*art_aP_copr['3M4'] +0.4796 #from NDVI & aP regression 

idx=pd.date_range('01-01-2012','31-01-2019',freq='D')

#art_ndvi_copr.drop(art_ndvi_copr.tail(1).index,inplace=True)

art_ndvi_copr=art_ndvi_copr.set_index(pd.DatetimeIndex(idx)) #reset index to start in January 2012-January 2019
art_ndvi_copr[art_ndvi_copr > 0.8] = 0.75
art_ndvi_copr=art_ndvi_copr.loc['2012':'2018']

#%%artifical Kc from ndvi

a=0.7
b=0.1 
art_ndvi_copr['Kc1']= (1 - (a - art_ndvi_copr['ndvi1']) / (a - b)) #from NDVI & Net P regression
art_ndvi_copr['Kc2']= 1 - (a - art_ndvi_copr['ndvi2']) / (a - b)
art_ndvi_copr['Kc3']= 1 - (a - art_ndvi_copr['ndvi3']) / (a - b)
art_ndvi_copr['Kc4']= 1 - (a - art_ndvi_copr['ndvi4']) / (a - b)



#%%
data_art = data_art.loc['2012-01-01':'2018-12-31']
#Names of columns that we want
varnames = ['Rain','P1','P2','PET', 'P3','PET_art']
#Extract numpy arrays from variables in dataframe
series = [column.values for label, column in data_art[varnames].items()]
#Make separate names for each - make new np array for calc
Rain, P1, P2, RET, P3,RET_art= series


N = len(data_art.index) #Number of timesteps in simulation


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
fc=0.42
wp=0.12
Pc=0.4
Zr=775

#%% calculate other parameters
TAW =  (fc - wp) * Zr #total available water (mm)
RAW = Pc * TAW               #readily available water (mm)
P = P2#Use P1,P2,P3,Rain
PET = RET * art_ndvi_copr['Kc2']
               #PET=AET in the model #artifical kc varies with each scenario 1,2,3
#PET = RET_art *art_ndvi_copr['Kc3']     #for artifical PET scenario         
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
        
        Theta[t] = fc - SMD[t] / Zr #to calculate SM from SMD
        if t>0:
            MB[t] = P[t-1] - AET[t-1] - D[t-1] + SMD[t] - SMD[t-1]
    
#%% Modelled SM into Df - the model ran from october 2007 but cut down here for NSE to start in 2008
SM_mod=pd.DataFrame(Theta)

SM_mod.drop(SM_mod.tail(1).index,inplace=True)
SM_mod = SM_mod.set_index(data_art.index)
SM_art2_test= SM_mod

#%%

D = pd.DataFrame(D)
D.drop(D.tail(1).index,inplace=True)
D2_Test= D.set_index(pd.DatetimeIndex(data_art.index))

AET = pd.DataFrame(AET)
AET.drop(AET.tail(1).index,inplace=True)
AET3=AET.set_index(pd.DatetimeIndex(data_art.index))


#%% create a mean SMD and mean SM of all the MCs to plot over 2 years
plt.subplot(211)
plt.plot(SM_art1)
plt.twinx()
plt.bar(data_art.index,data_art['P1'],width=10, color='black', alpha =0.4, align='center')

plt.subplot(212)
plt.plot(D1)


#%%
data_art=data_art.resample('M').sum()
data_art_a=data_art_a.resample('M').sum()

SM_art1=SM_art1.resample('M').mean()
SM_art2=SM_art2.resample('M').mean()
SM_art3=SM_art3.resample('M').mean()
SM_art4=SM_art4.resample('M').mean()

SM_art_air1=SM_art_air1.resample('M').mean()
SM_art_air2=SM_art_air2.resample('M').mean()
SM_art_air3=SM_art_air3.resample('M').mean()
SM_art_air4=SM_art_air4.resample('M').mean()

por1=0.710
por2=0.34


t=data_art.index #starts from 2012 January (becasue NDVI is from january too and as a result SM)
diff1 = data_art.P2 - data_art.P1
diff2 = data_art.P1 - data_art.Rain
diff3= data_art.P3 -data_art.P1
months = mdates.MonthLocator()
years = mdates.YearLocator()
x_labels=['Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8']
years=(0,1,2,3,4,5,6,7)

t_air=data_art_a.index
diff_air1 = data_art_a.P2 - data_art_a.P1
diff_air2 = data_art_a.P1 - data_art_a.Rain
diff_air3= data_art_a.P3 -data_art_a.P1


#%% Plot Scenarios 

plt.style.use('default')
fig = plt.figure() 
ax1=fig.add_subplot(421) #historic COP

ax1.plot(SM_art4[0]/por1*100, color='blue') #historic with Netp as Lead indicator   
ax1.set_ylabel('Saturation (%)', fontsize = 'small', fontweight='bold')
ax1.tick_params(axis= 'both', which ='major', labelsize='8')
#ax1.yaxis.set_ticks(np.arange(0,0.8,0.1))
ax1.set_ylim(0,100)
ax1.xaxis.set_minor_locator(months)
#ax1.set_xticklabels(x_labels)
ax1.axhline(0.17/por1*100, color='black', alpha = 0.6)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.invert_yaxis()
ax2.bar(t, data_art.Rain, color='black', width = 20, alpha =0.6, label = 'Precipitation')
#ax2.set_ylabel('P (mm/month)', fontsize = 'small',fontweight='bold')  # we already handled the x-label with ax1
ax2.tick_params(axis='y', which = 'major', labelsize ='8')
ax2.set_ylim(250,0)


ax3 = fig.add_subplot(422) #Historic AIRS
ax3.plot(SM_art_air4[0]/por2*100, color='orange')   
ax3.tick_params(axis= 'both', which ='major', labelsize='8')
ax3.set_ylim(0,100)
ax3.axhline(0.07/por2*100, color='black', alpha = 0.5)
ax3.xaxis.set_minor_locator(months)
#ax3.yaxis.set_ticks(np.arange(0,0.8,0.1))

ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
ax4.invert_yaxis()
ax4.bar(t_air, data_art_a.Rain, color='black', width = 20, alpha = 0.6, label = 'Precipitation')
ax4.set_ylabel('P (mm/month)', fontsize = 'small',fontweight='bold')  # we already handled the x-label with ax1
ax4.tick_params(axis='y', which = 'major', labelsize ='8')
ax4.set_ylim(250,0)


ax5=fig.add_subplot(423) #Scneario 1 COPR
ax5.set_ylabel('Saturation (%)', fontsize = 'small',fontweight='bold')
ax5.plot((SM_art1[0]/por1)*100, color = 'blue')
ax5.tick_params(axis= 'both', which ='major', labelsize='8')
#ax5.yaxis.set_ticks(np.arange(0,0.8,0.1))
ax5.set_ylim(0,100)
ax5.xaxis.set_minor_locator(months)
ax5.axhline(0.17/por1*100, color='black', alpha = 0.6)

ax6 = ax5.twinx()  # instantiate a second axes that shares the same x-axis
ax6.invert_yaxis()
ax6.bar(t, data_art.P1, color='black', width = 20, alpha = 0.6, label = 'Precipitation')
ax6.bar(t, diff2, color='red', bottom = data_art.Rain, width = 20,alpha=0.4,label = 'Precipitation')
#ax6.set_ylabel('P (mm/month)', fontsize = 'small',fontweight='bold')  # we already handled the x-label with ax1
ax6.tick_params(axis='y', which = 'major', labelsize ='8')
ax6.set_ylim(250,0)


ax7 = fig.add_subplot(424) ##Scenario 1 AIRS
#ax7.set_ylabel('Saturation (%)', fontsize = 'small',fontweight='bold')
ax7.plot((SM_art_air1[0]/por2)*100, color = 'orange')
ax7.tick_params(axis= 'both', which ='major', labelsize='8')
ax7.set_ylim(0,100)
ax7.axhline(0.07/por2*100, color='black', alpha = 0.5)
ax7.xaxis.set_minor_locator(months)
#ax7.yaxis.set_ticks(np.arange(0,0.8,0.1))

ax8 = ax7.twinx()  # instantiate a second axes that shares the same x-axis
ax8.invert_yaxis()
ax8.bar(t_air, data_art_a.P1, color='black', width = 20, alpha = 0.6, label = 'Precipitation')
ax8.bar(t_air, diff_air2, color='red', bottom = data_art_a.Rain, width = 20,alpha=0.3, label = 'Precipitation')
ax8.set_ylabel('P (mm/month)', fontsize = 'small',fontweight='bold')  # we already handled the x-label with ax1
ax8.tick_params(axis='y', which = 'major', labelsize ='8')
ax8.set_ylim(250,0)


ax9 = fig.add_subplot(425) #Scenario 2 COPR
ax9.set_ylabel('Saturation (%)', fontsize = 'small',fontweight='bold')
ax9.plot(SM_art2[0]/por1*100, color = 'blue')
ax9.tick_params(axis= 'both', which ='major', labelsize='8')
ax9.set_ylim(0,100)
ax9.xaxis.set_minor_locator(months)
#ax9.yaxis.set_ticks(np.arange(0,0.8,0.1))
ax9.axhline(0.17/por1*100,color='black', alpha = 0.6)

ax10 = ax9.twinx()  # instantiate a second axes that shares the same x-axis
ax10.invert_yaxis()
ax10.bar(t, data_art.P2, color='black', width = 20, alpha = 0.6, label = 'Precipitation')
ax10.bar(t, diff1, color='green', bottom = data_art.P1, width = 20,label = 'Precipitation')
ax10.bar(t, diff2, color='red', bottom = data_art.Rain, alpha=0.4,width = 20,label = 'Precipitation')
#ax10.set_ylabel('P (mm/month)', fontsize = 'small',fontweight='bold')  # we already handled the x-label with ax1
ax10.tick_params(axis='y', which = 'major', labelsize ='8')
ax10.set_ylim(250,0)

ax11 = fig.add_subplot(426) #Scenario 2 AIRS
#ax11.set_ylabel('Saturation (%)', fontsize = 'small',fontweight='bold')
ax11.plot(SM_art_air2[0]/por2*100, color = 'orange')
ax11.tick_params(axis= 'both', which ='major', labelsize='8')
ax11.axhline(0.07/por2*100, color='black', alpha = 0.5)
ax11.set_ylim(0,100)
ax11.xaxis.set_minor_locator(months)
#ax11.yaxis.set_ticks(np.arange(0,0.8,0.1))

ax12 = ax11.twinx()  # instantiate a second axes that shares the same x-axis
ax12.invert_yaxis()
ax12.bar(t_air, data_art_a.P2, color='black', width = 20, alpha = 0.6, label = 'Precipitation')
ax12.bar(t_air, diff_air1, color='green', bottom = data_art_a.P1, width = 20,label = 'Precipitation')
ax12.bar(t_air, diff_air2, color='red', bottom = data_art_a.Rain, width = 20,alpha=0.3,label = 'Precipitation')
ax12.set_ylabel('P (mm/month)', fontsize = 'small',fontweight='bold')  # we already handled the x-label with ax1
ax12.tick_params(axis='y', which = 'major', labelsize ='8')
ax12.set_ylim(250,0)


ax13 = fig.add_subplot(427) #Scenario 3 COPR
ax13.set_ylabel('Saturation (%)', fontsize = 'small',fontweight='bold')
ax13.plot(SM_art3[0]/por1*100, color = 'blue')
ax13.tick_params(axis= 'both', which ='major', labelsize='8')
ax13.set_ylim(0,100)
ax13.set_xlabel('Date', fontsize = 'small', fontweight='bold')
ax13.xaxis.set_minor_locator(months)
#ax13.yaxis.set_ticks(np.arange(0,0.8,0.1))
ax13.axhline(0.17/por1*100, color='black', alpha = 0.6)

ax14 = ax13.twinx()  # instantiate a second axes that shares the same x-axis
ax14.invert_yaxis()
ax14.bar(t, data_art.P3, color='black', width = 20, alpha = 0.6, label = 'Precipitation')
ax14.bar(t, diff2, color='red', bottom = data_art.Rain,alpha=0.4, width = 20,label = 'Precipitation')
ax14.bar(t, diff3, color='red', bottom = data_art.Rain, alpha=0.4,width = 20,label = 'Precipitation')
ax14.tick_params(axis='y', which = 'major', labelsize ='8')
ax14.set_ylim(250,0)

ax15 = fig.add_subplot(428) #Scenario 3 COPR
#ax15.set_ylabel('Saturation (%)', fontsize = 'small',fontweight='bold')
ax15.plot(SM_art_air3[0]/por2*100, color = 'orange')
ax15.tick_params(axis= 'both', which ='major', labelsize='8')
ax15.axhline(0.07/por2*100, color='black', alpha = 0.5)
ax15.set_ylim(0,100)
ax15.set_xlabel('Date', fontsize = 'small', fontweight='bold')
ax15.xaxis.set_minor_locator(months)
#ax15.yaxis.set_ticks(np.arange(0,0.8,0.1))

ax16 = ax15.twinx()  # instantiate a second axes that shares the same x-axis
ax16.invert_yaxis()
ax16.bar(t_air, data_art_a.P3, color='black', width = 20, alpha = 0.6, label = 'Precipitation')
ax16.bar(t_air, diff_air2, color='red', bottom = data_art_a.Rain, alpha=0.4,width = 20,label = 'Precipitation')
ax16.bar(t_air, diff_air3, color='red', bottom = data_art_a.Rain, alpha=0.4,width = 20,label = 'Precipitation')
ax16.set_xlabel('Year', fontsize = 'small', fontweight='bold')
#ax16.set_ylabel('P (mm/month)', fontsize = 'small',fontweight='bold')  # we already handled the x-label with ax1
ax16.tick_params(axis='y', which = 'major', labelsize ='8')
ax16.set_ylim(250,0)

plt.show()

#%%Plot cumulative sum so f variables, D, SMD and AET
fig=plt.figure()
plt.style.use('bmh')
plt.subplot(231)
plt.plot(data_art['P1'].cumsum(),label='Scenario A')
plt.plot(data_art['P2'].cumsum(),label='Scenario B')
plt.plot(data_art['P3'].cumsum(),label='Scenario C')
plt.plot(data_art['Rain'].cumsum(),label='Historic')
plt.subplot(232)
plt.plot(AET1[0].cumsum())
plt.plot(AET2[0].cumsum())
plt.plot(AET3[0].cumsum())
plt.plot(AET4[0].cumsum())
#plt.legend(frameon=False,fontsize='small', loc='best')

plt.subplot(233)
plt.plot(D1[0].cumsum())
plt.plot(D2.cumsum())
plt.plot(D3.cumsum())
plt.plot(D4.cumsum())
#plt.legend(frameon=False,fontsize='small', loc='best')
#plt.ylim(0,230)

plt.subplot(234)
plt.plot(data_art_a['P1'].cumsum())
plt.plot(data_art_a['P2'].cumsum())
plt.plot(data_art_a['P3'].cumsum())
plt.plot(data_art_a['Rain'].cumsum())
#plt.legend(frameon=False,fontsize='small', loc='best')

plt.subplot(235)
plt.plot(AET1_a[0].cumsum())
plt.plot(AET2_a[0].cumsum())
plt.plot(AET3_a[0].cumsum())
plt.plot(AET4[0].cumsum())
#plt.legend(frameon=False,fontsize='small', loc='best')

plt.subplot(236)
plt.plot(D1_a[0].cumsum())
plt.plot(D2_a[0].cumsum())
plt.plot(D3_a[0].cumsum())
plt.plot(D4_a[0].cumsum())

fig.legend(fontsize='x-small', ncol= 4, loc ='lower center', frameon=False)
#plt.ylim(0,230)
#%% Time below the stress threshold 













