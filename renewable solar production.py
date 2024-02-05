#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import pandas_profiling
import seaborn as sns
import datetime as dt

import xgboost as xgb# for XGBoost algorithm
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# Set the default plot size
plt.rcParams['figure.figsize'] = [15, 8]
# set a grid for each plot too
sns.set_style("whitegrid")


# In[2]:


df = pd.read_csv("C:\\Users\\20109\\Downloads\\solar_weather.csv\\solar_weather.csv", index_col='Time', parse_dates=True)


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.index.name


# In[6]:


df.index.dtype


# In[7]:


# desciption of the data
df.describe()


# In[8]:


# Any nulls - no
df.isna().sum()


# In[9]:


# Create a 4x4 grid of subplots
fig, axes = plt.subplots(4, 4, figsize=(15, 15))
colours = "bgrcmykbgrcmykbg"

# Flatten the axes for easy iteration
axes = axes.flatten()

# Iterate through the columns and create separate plots
for i, col in enumerate(df.columns):
    sns.histplot(df[col], ax=axes[i], color=colours[i])  


# In[10]:


# Boxplots of all the data
colours = "bgrcmykbgrcmykbg"
i=0
for column in df:    
    fig,ax = plt.subplots(figsize=(8,0.2))
    sns.boxplot(data=df, x=column, color= colours[i]) 
    i+=1


# In[11]:


avg_energy = df.groupby("weather_type")["Energy delta[Wh]"].mean()
plt.bar(avg_energy.index, avg_energy.values)
plt.title("Average Energy Production by Weather Condition")
plt.xlabel("Weather Condition")
plt.ylabel("Average Energy Production")
plt.show()


# In[12]:


plt.scatter(df["GHI"], df["Energy delta[Wh]"])
plt.title("Energy Consumption vs Solar Radiation")
plt.xlabel("Solar Radiation")
plt.ylabel("Energy Consumption")
plt.show()


# In[13]:


#plot the average energy production per month 
avg_energy = df.groupby("month")["Energy delta[Wh]"].mean()
plt.plot(avg_energy.index, avg_energy.values)
plt.title("Average Energy Production by Month")
plt.xlabel("Month")
plt.ylabel("Average Energy Production")
plt.show()


# In[14]:


df_hour = df.groupby('hour').mean(numeric_only=False)
sns.barplot(data=df_hour, x=df_hour.index, y='Energy delta[Wh]')
plt.title('Energy Delta by Hour', fontsize=24)


# In[15]:


# Now to visualise amount of EDnergy Delta by hour
sns.boxplot(x=df.index.hour, y='Energy delta[Wh]', data=df) 
plt.ylabel('Energy delta[Wh]', fontsize=24)
plt.xlabel('Hour', fontsize=24)
plt.title("Range of Energy Delta values per hour", fontsize=34)


# In[16]:


# More energy is used in warmer temperatures indicating greater use of air conditioners to cool rather than heaters to warm.
fig, ax = plt.subplots(1, 1, figsize=(15,8))

df_temp = df.groupby('temp').mean() 
sns.barplot(data=df_temp, x=df_temp.index, y='Energy delta[Wh]')

# only put labels every 20th label - roughly every 2 degrees - saves x axis being so cluttered
for i, label in enumerate(ax.get_xticklabels()):
    if i % 20 != 0:  # Display every twentieth label
        label.set_visible(False)


# In[17]:


# get correlations for all columns 
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")


# In[18]:


#Based on the correlations above a model will be developed to see if we can forecast energy usage based on all the features (predictors)
# form the correlation variable created above let's get the top 3 (using absolute value to get either positive or negative) correlations with Energy delta
corr[['Energy delta[Wh]']].abs().sort_values(by='Energy delta[Wh]', axis=0, ascending=False)


# In[19]:


#Predictions of Energy delta based using XGBoost
df.index.min(),df.index.max()


# In[20]:


#break up the data into training/testing
# all columns except Energy Delta - the predictors
X = df.drop('Energy delta[Wh]', axis=1)
# Energy Delta - the target variable
y = df['Energy delta[Wh]']

# use stratify by montth to make sure we get the same number of days for each month - so the data doesn't get skewed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=X.index.month)
print('# train X:',len(X_train),'# test X:',len(X_test),'# train y:',len(y_train),'# test y:',len(y_test))


# In[21]:


regr = XGBRegressor(objective ='reg:squarederror') # this is the defualt objective but i played with other - this gave the best result
regr.fit(X_train,y_train)
y_pred = regr.predict(X_test)
# store the original and predicted value in a dataframe
df_final = pd.DataFrame(data={'Predictions':y_pred, 'Actuals':y_test }) 

# the model gives some negative predictions which doesn't make sense so set them to 0
print(df_final[df_final['Predictions']<0].head())
# set these negative values to 0
df_final['Predictions'][df_final['Predictions']<0] = 0
# print them again just to confirm
print(df_final[df_final['Predictions']==0].head())
# zoom in to 300 values - note due to stratification the samples are random and the whole timeseries appears
df_final.plot(alpha=0.5) # reduce opacity to see both lines
plt.title('XGBRegressor Actuals vs Prediction', fontsize=30)
plt.ylabel('Energy Delta')
print('\nAccuragy for XGBRegressor                                     = {:.2f}'.format(regr.score(X_test, y_test)))    
print('The Coefficient of determination (R-squared) for XGBRegressor = {:.2f}'.format(r2_score(df_final['Actuals'],df_final['Predictions'])))
print('The mean absolute error (MAE) for XGBRegressor                = {:.2f}'.format(mean_absolute_error( df_final['Actuals'],df_final['Predictions'])))
print('The RMSE error (RMSE) for XGBRegressor                        = {:.2f}'.format(mean_squared_error( df_final['Actuals'],df_final['Predictions']), squared=True))


# In[22]:


#look at some more algorithms to see if XGBoost is the best or can be improved upon
from sklearn import linear_model 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

def regression_calculations(X_train, y_train, X_test, y_test, regressor):
    
    regr = regressor.fit(X_train,y_train)
    y_pred = regr.predict(X_test)
    # store the orignial and predicted value in a dataframe
    df_final = pd.DataFrame(data={'Predictions':y_pred, 'Actuals':y_test })      
    df_final.plot(alpha=0.5) # reduce opacity to see both lines
    plt.title(regressor, fontsize=30)
    plt.ylabel('Energy Delta')
    print('Accuracy for                                     ',regressor, ' = {:.4f}'.format(regr.score(X_test, y_test)))    
    print('The Coefficient of determination (R-squared) for ',regressor, ' = {:.2f}'.format(r2_score(df_final['Actuals'],df_final['Predictions'])))
    print('The mean absolute error (MAE) for                ',regressor, ' = {:.2f}'.format(mean_absolute_error( df_final['Actuals'],df_final['Predictions'])))
    print('The RMSE error (RMSE) for                        ',regressor, ' = {:.2f}'.format(mean_squared_error( df_final['Actuals'],df_final['Predictions']), squared=False))
   
    
    print('\n\n')
  


# In[23]:


# define the different regressors
regressors = [LinearRegression(), DecisionTreeRegressor(), linear_model.Lasso(), GradientBoostingRegressor()] 


# In[24]:


#loop through the different regressors
for regressor in regressors:
    pred = regression_calculations(X_train, y_train, X_test, y_test, regressor)


# In[27]:


def get_user_input_and_predict(df, model):
    temperature = float(input("Enter the temperature: "))
    pressure = float(input("Enter the pressure: "))
    hour = float(input("Enter the hour: "))
    humidity = float(input("Enter the humidity: "))
    wind_speed = float(input("Enter the wind speed: "))
    
    user_input = pd.DataFrame(data={'temp': [temperature], 'pressure': [pressure], 'hour': [hour], 'humidity': [humidity], 'wind_speed': [wind_speed]})
    
    # Use lowercase names when selecting columns
    user_input = user_input[['temp', 'pressure', 'hour', 'humidity', 'wind_speed']]
    
    # Append the user input to the existing DataFrame 'df'
    df_with_user_input = pd.concat([df, user_input], ignore_index=True)
    
    # Use the model to predict energy consumption for the user input
    energy_prediction = model.predict(df_with_user_input.tail(1).drop('Energy delta[Wh]', axis=1))

    # Display the prediction
    print(f'Predicted Energy Consumption: {energy_prediction[0]} Wh')

    return df_with_user_input

# Get user input, append it to the existing DataFrame 'df', and get a prediction
df = get_user_input_and_predict(df, regr)


# In[28]:


#Save the model 
import joblib 
joblib.dump(regr,'renewable.h5')


# In[29]:


loaded_nb_model = joblib.load('renewable.h5')


# In[ ]:




