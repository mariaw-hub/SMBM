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
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
from scipy.stats import powerlaw
from sklearn.preprocessing import PolynomialFeatures
import operator 

#%%
aP = pd.read_csv("Infiltration_COPR180221.csv", sep=';') #COPR

aP = pd.read_csv("Infiltration_AIR180221.csv", sep='\t') #AIRS


aP['Date'] = pd.to_datetime(aP['Date'], utc=True)
aP=aP.set_index(pd.DatetimeIndex(aP['Date'])) 
#aP = aP.resample('M').sum() keeping it daily per 09.03.21
aP['Year'] = aP.index.year
aP['Net_P'] = aP['Rain_mm_Tot']- aP['PET']


aP['3M'] = aP.iloc[:,4].rolling(window='90D').sum() #and 3 month lag for COPR 

aP['3M'] = aP.iloc[:,4].rolling(window='60D').sum() #2month lag for AIRS 


aP_copr = pd.DataFrame(aP.loc['2007-12-01':'2019-08-31']) # 2 month lag for COPR
aP_copr.drop(aP_copr.tail(1).index,inplace=True)
ndvi_copr_aP = ndvi_copr.loc['2008-01-01':'2019-09-30']


aP_airs = pd.DataFrame(aP.loc['2008-01-01':'2019-08-31']) # 1 month lag for AIRS                                   
aP_airs.drop(aP_airs.tail(1).index,inplace=True)
ndvi_airs_aP = ndvi_airs.loc['2008-02-01':'2019-09-30']


#%%

#%%
#Determine relationship between 3 monthly Rain and NDVI
stats.pearsonr(aP_copr['3M'], ndvi_copr_aP['median']) 
 #corr=0.84 with Net P 23.2.21

#AIRS
stats.pearsonr(aP_airs['3M'], ndvi_airs_aP['median']) 
#corr=0.74 with Net P 23.2.21
#%%
#Reset Index to  start in JAnuary(copr)/February(airs) 2008 (NDVI)
idx=pd.date_range('01-01-2008','09-30-2019',freq='D') #daily daterange

#reset index for aP and ndvi to start in January 2008 for COPR
aP_copr=aP_copr.set_index(pd.DatetimeIndex(idx))
ndvi_copr_aP=ndvi_copr_aP.set_index(pd.DatetimeIndex(idx))


#reset index for aP and ndvi to start in february 2008 for AIRS
idx=pd.date_range('02-01-2008','09-30-2019',freq='D') #daily daterange
aP_airs=aP_airs.set_index(pd.DatetimeIndex(idx))
ndvi_airs_aP=ndvi_airs_aP.set_index(pd.DatetimeIndex(idx))


#%%
#Include only NDVI > 0.3 to fit regression 
aP_copr=pd.concat([aP_copr['3M'], ndvi_copr_aP['median']], axis=1) #combine df
aP_airs=pd.concat([aP_airs['3M'], ndvi_airs_aP['median']], axis=1) #AIRS combine df

#Threshold against saturation
#aP_copr=aP_copr.drop(aP_copr[aP_copr['median'] > 0.7].index) 
#aP_airs=aP_airs.drop(aP_airs[aP_airs['median'] > 0.7].index) 

aP_copr = aP_copr.drop(aP_copr[aP_copr['3M'] > 100].index) #to exlude outliers of netP during D 
aP_airs = aP_airs.drop(aP_airs[aP_airs['3M'] > 100].index) #to exclude outliers of nDVi at higher NetP during D
#%%


#%%
plt.subplot(211)
plt.scatter(aP_copr['3M'],aP_copr['median'],color='blue')
plt.subplot(212)
plt.scatter(aP_airs['3M'],aP_airs['median'],color='orange')



#%% plot Regression between NEtp and NDVI

x=np.asarray(aP_copr['3M'])
y=np.asarray(aP_copr['median'])

# transforming the data to include another axis
x = x[:, np.newaxis]
y = y[:, np.newaxis]

polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

reg_label = "Inliers coef:%s - b:%0.2f" % \
    (np.array2string(model.coef_,
                     formatter={'float_kind': lambda fk: "%.3f" % fk}),
     model.intercept_)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print(rmse)
print(r2)

plt.scatter(x, y, s=10)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='r')
#plt.xlabel('FVC')
#plt.ylabel('INT (mm/month)')
plt.show()
#plt.text(0.3,150,'y=-52.97-246.44*x*324.31*x^2, R^2=0.89')
print(reg_label)
#plt.text(0.15,1.67, 'R^2=0.71')







#%% Regression for COPR

slope, intercept, r_value, p_value, std_err = stats.linregress(aP_copr['3M'],aP_copr['median'])
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(aP_airs['3M'],aP_airs['median'])

plt.subplot(211)
sns.regplot(aP_copr['3M'], aP_copr['median'], data=aP_copr, color='blue',
      ci=None, label="y={0:.4f}x+{1:.4f}".format(slope, intercept)).legend(loc="best", frameon=False)
#pearson R = 0.82 10.3.21

plt.subplot(212)
#Regression for Airs
sns.regplot(aP_airs['3M'], aP_airs['median'], data=aP_airs,color='orange', 
      ci=None, label="y={0:.4f}x+{1:.4f}".format(slope1, intercept1)).legend(loc="best", frameon=False)

#Pearson R=0.76 10.3.21









