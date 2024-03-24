#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv("26th_December_2023_Air_Quality_Dataset.csv")


# In[5]:


data.info()


# In[6]:


data.head(3)


# In[7]:


numerical_data = data.select_dtypes(include=np.number)
categorical_data = data.select_dtypes(exclude=np.number)


# In[8]:


numerical_data = numerical_data.drop(["id"], axis=1)
numerical_data


# In[9]:


categorical_data = categorical_data.drop(["country", "last_update"], axis=1)
categorical_data


# In[10]:


data.describe()


# In[11]:


for k in categorical_data:
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)

    sns.histplot(categorical_data[k],kde=True)        
    plt.tight_layout()
    plt.show()


# In[12]:


city_complaints_freq = data.groupby(['city'])['pollutant_avg'].size().sort_values(ascending=False)
city_complaints_freq = city_complaints_freq.head(5)
plt.figure(figsize=(10, 6))  # Set the figure size
city_complaints_freq.plot.bar(rot=90, fontsize=10, figsize=(10, 6), color='red',  title= "TOp 5 polluted cities")
plt.xlabel('City')
plt.ylabel('Count')
plt.show()


# In[13]:


city_list = []

# for city, freq in city_complaints_freq.items():
#     print(f"City: {city}, Frequency: {freq}")
    
for i in range(len(city_complaints_freq)):
    city = city_complaints_freq.index[i]
    freq = city_complaints_freq.iloc[i]
    city_list.append(city) 
city_list    


# In[14]:


plt.figure(figsize=(12, 8))
plt.scatter(data['longitude'], data['latitude'], alpha=0.5, s=20, c='blue')
plt.title('Latitude and Longitude for All Cities')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()


# In[15]:


city_data = data[data['city'].isin(city_list)]
for city in city_list:
    city_subset = city_data[city_data['city'] == city]

    
    plt.figure(figsize=(10, 10))
    plt.hexbin(x=city_subset['longitude'], y=city_subset['latitude'], gridsize=50, cmap='flare')

    
    plt.title('Pollution in city ' + city)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.show()


# In[16]:


from scipy.stats import skew, kurtosis
from scipy.stats import probplot

skew_pol_min = skew(numerical_data['pollutant_min'])
k_pol_min = kurtosis(numerical_data['pollutant_min'])
print("Skweness and kurtosis of pollutant_min",skew_pol_min, k_pol_min)

skew_pol_max = skew(numerical_data['pollutant_max'])
k_pol_max = kurtosis(numerical_data['pollutant_max'])
print("Skweness and kurtosis of pollutant_max",skew_pol_max, k_pol_max)

k_pol_avg = kurtosis(numerical_data['pollutant_avg'])
skew_pol_avg = skew(numerical_data['pollutant_avg'])
print("Skweness and kurtosis of pollutant_avg",skew_pol_avg, k_pol_avg)


# In[17]:


data_kurt = numerical_data[["pollutant_max","pollutant_min","pollutant_avg"]]

for i in data_kurt:
    
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(data_kurt[i],kde=True,color='green', bins=10)  
    plt.title("Data hist  "+i)
    plt.ylabel("Freq")
    plt.xlabel(i)

# Find the mean, median, mode
    mean_price = data_kurt[i].mean()
    median_price = data_kurt[i].median()
    mode_price = data_kurt[i].mode().squeeze()

# Add vertical lines at the position of mean, median, mode
    plt.axvline(mean_price, label="Mean")
    plt.axvline(median_price, color="black", label="Median")
    plt.axvline(mode_price, color="green", label="Mode")
    plt.legend()
    
    
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data_kurt[i])
    plt.title("Data boxplot  "+i)



#    sns.kdeplot(data_kurt[i], color="red")
#    sns.despine(top=True, right=True, left=True)
#    plt.xticks([])
#    plt.yticks([])




    plt.show()


# In[18]:


plt.figure(figsize=(8, 6))
correlation_matrix = numerical_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix of numerical data')
plt.show()


# In[19]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
for column in categorical_data.columns:
    if categorical_data[column].dtype == 'object':
        categorical_data[column] = label_encoder.fit_transform(categorical_data[column])

# Display the DataFrame after label encoding
print("\nDataFrame after Label Encoding:")
print(categorical_data)


# In[20]:


plt.figure(figsize=(8, 6))
correlation_matrix = categorical_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix of categorical data')
plt.show()


# In[21]:


combined_data = pd.concat([numerical_data, categorical_data], axis=1)
plt.figure(figsize=(8, 6))
correlation_matrix = combined_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix of Air quality index')
plt.show()


# In[22]:


data_new = combined_data.drop(columns=["longitude", "latitude","state"])
data_new


# In[23]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(data_frame):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data_frame.columns
    vif_data["VIF"] = [variance_inflation_factor(data_frame.values, i) for i in range(data_frame.shape[1])]
    return vif_data

vif_results1 = calculate_vif(combined_data)
high_vif1 = vif_results1[vif_results1['VIF'] > 10]

vif_results2 = calculate_vif(data_new)
high_vif2 = vif_results2[vif_results2['VIF'] > 10]

print("Features with high VIF for the complete dataset",high_vif1 )
print("\n")
print("Features with high VIF for dataset with features removed after correlation matrix observation",high_vif2 )


# In[24]:


y_original = data_new.pop('pollutant_avg')
x_original = data_new
print(y_original)
print("\n")
print(x_original)


# In[25]:


from sklearn.model_selection import train_test_split
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(x_original, y_original, test_size=0.2, random_state=42)


# In[26]:


from sklearn.feature_selection import SelectKBest, f_regression

k = 3
selector = SelectKBest(score_func=f_regression, k=k)
selector.fit(X_train_f, y_train_f)

selected_feature_names = X_train_f.columns[selector.get_support()]
print("Selected Features:", selected_feature_names)


# In[27]:


from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

model = LinearRegression()
rfecv = RFECV(estimator=model, cv=5)  # You can specify the number of cross-validation folds
rfecv.fit(X_train_f, y_train_f)

selected_feature_names = X_train_f.columns[rfecv.support_]
print("Selected Features:", selected_feature_names)


# In[28]:


x_original_new = data_new[["pollutant_min","pollutant_max","station_code","pollutant_id"]]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_original_new, y_original, test_size=0.2, random_state=42)


# In[29]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[30]:


models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Huber Regression': HuberRegressor(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

# Train models and evaluate performance
results = {'Model': [], 'MSE': [], 'MAE': [], 'R^2': []}
predictions = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results['Model'].append(model_name)
    results['MSE'].append(mse)
    results['MAE'].append(mae)
    results['R^2'].append(r2)

    predictions[model_name] = y_pred
    
    
# Display results
results_df = pd.DataFrame(results)
print(results_df)

# Plotting actual vs. predicted values
plt.figure(figsize=(12, 6))

for model_name, y_pred in predictions.items():
    plt.scatter(y_test, y_pred, label=model_name)

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='black', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()


# In[31]:


test = X_test.head(5)
predy = y_test.head(5).to_numpy()
models['Random Forest'].fit(X_train, y_train)

for i in range(len(test)):
    pol_avg_pred = models['Random Forest'].predict(test)
    print("Actual and predicted values \n",predy[i],"\t",pol_avg_pred[i])
    print("\n")


# In[32]:


import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[ ]:




