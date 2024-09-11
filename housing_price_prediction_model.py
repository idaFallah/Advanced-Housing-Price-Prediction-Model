
#data preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('/content/train.csv')

#exploring the data

dataset.head()

dataset.shape

dataset.columns

dataset.info()

#statistical summary
dataset.describe()

#numerical columns
dataset.select_dtypes(include=['int64', 'float64']).columns

len(dataset.select_dtypes(include=['int64', 'float64']).columns)

#categorical columns
dataset.select_dtypes(include='object').columns

len(dataset.select_dtypes(include='object').columns)

#dealing with null values
dataset.isnull().values.any()

dataset.isnull().values.sum()

dataset.columns[dataset.isnull().any()] #columns with null values

len(dataset.columns[dataset.isnull().any()])

#show null values with heatmap
plt.figure(figsize=(16, 9))
sns.heatmap(dataset.isnull())
plt.show()

dataset.shape #19 of 81 has null values

null_precent = dataset.isnull().sum() / dataset.shape[0] *100

null_precent

#columns to drop
cols_to_drop = null_precent[null_precent > 50].keys()

cols_to_drop

dataset = dataset.drop(columns=['Alley', 'MasVnrType', 'PoolQC', 'Fence', 'MiscFeature'])

dataset.shape

dataset.columns[dataset.isnull().any()] #columns with null values

len(dataset.columns[dataset.isnull().any()])

#add columns mean to numerical columns to fill in the null values

#numerical columns : 'LotFrontage', 'MasVnrArea', 'GarageYrBlt'

dataset['LotFrontage'] = dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean())
dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mean())
dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(dataset['GarageYrBlt'].mean())

len(dataset.columns[dataset.isnull().any()])

#add columns mode to categorical columns to fill in the null values

dataset.select_dtypes(include='object').columns

dataset.columns[dataset.isnull().any()]

len(dataset.columns[dataset.isnull().any()])

dataset['BsmtQual'] = dataset['BsmtQual'].fillna(dataset['BsmtQual'].mode()[0])
dataset['BsmtCond'] = dataset['BsmtCond'].fillna(dataset['BsmtCond'].mode()[0])
dataset['BsmtExposure'] = dataset['BsmtExposure'].fillna(dataset['BsmtExposure'].mode()[0])
dataset['BsmtFinType1'] = dataset['BsmtFinType1'].fillna(dataset['BsmtFinType1'].mode()[0])
dataset['BsmtFinType2'] = dataset['BsmtFinType2'].fillna(dataset['BsmtFinType2'].mode()[0])
dataset['Electrical'] = dataset['Electrical'].fillna(dataset['Electrical'].mode()[0])
dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna(dataset['FireplaceQu'].mode()[0])
dataset['GarageType'] = dataset['GarageType'].fillna(dataset['GarageType'].mode()[0])
dataset['GarageFinish'] = dataset['GarageFinish'].fillna(dataset['GarageFinish'].mode()[0])
dataset['GarageQual'] = dataset['GarageQual'].fillna(dataset['GarageQual'].mode()[0])
dataset['GarageCond'] = dataset['GarageCond'].fillna(dataset['GarageCond'].mode()[0])

len(dataset.columns[dataset.isnull().any()])

dataset.isnull().values.any()

dataset.select_dtypes(include='object').columns

# distplot of the target variable

plt.figure(figsize=(16,9))
bar = sns.distplot(dataset['SalePrice'])
bar.legend(["Skewness: {:.2f}".format(dataset['SalePrice'].skew())])
plt.show()

dataset_2 = dataset.drop(columns=['SalePrice'])

dataset_2.shape

dataset.select_dtypes(include=['float64']).columns

dataset_2.select_dtypes(include=['object']).columns

categorical_cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
       'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive', 'SaleType', 'SaleCondition']

dataset_2_encoded = pd.get_dummies(dataset_2, columns=categorical_cols)

dataset_2_encoded.corrwith(dataset['SalePrice']).plot.bar(
    figsize = (16, 9), title = 'Correlated with SalePrice', grid=True
)

# heatmap
plt.figure(figsize=(25, 25))
ax = sns.heatmap(data=dataset.corr(), cmap='coolwarm', annot=True, linewidths=2)

high_corr = dataset.corr()

high_corr_features = high_corr.index[abs(high_corr['SalePrice']) > 0.5]

high_corr_features

len(high_corr_features)

# heatmap
plt.figure(figsize=(16, 9))
ax = sns.heatmap(data=dataset[high_corr_features].corr(), cmap='coolwarm', annot=True, linewidths=2)

dataset.select_dtypes(include='object').columns

len(dataset.select_dtypes(include='object').columns)

dataset.shape

#splitting the dataset

#independant variables/ matrix of features
x = dataset.drop(columns='SalePrice')

#target variable/ dependant feature
y = dataset['SalePrice']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train.shape

x_test.shape

y_train.shape

y_test.shape

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train

x_test

#building the model
#start froom multiple linear regression
from sklearn.linear_model import LinearRegression

regressor_multiLR = LinearRegression()
regressor_multiLR.fit(x_train, y_train)

y_pred = regressor_multiLR.predict(x_test)

from sklearn.metrics import r2_score

r2_score(y_test, y_pred)

#random forest regression
from sklearn.ensemble import RandomForestRegressor
regressor_RFR = RandomForestRegressor()
regressor_RFR.fit(x_train, y_train)

y_pred = regressor_RFR.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

#XG boost regression
from xgboost import XGBRFRegressor
regressor_XGBR = XGBRFRegressor()
regressor_XGBR.fit(x_train, y_train)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

#hyperparam tuning
#randomized search
from sklearn.model_selection import RandomizedSearchCV

parameters = {
    'n_estimators':[200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    'max_depth':[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf':[1, 2, 4],
    'max_features':['sqrt', None],
    'bootstrap':[True, False]
}

parameters

random_cv = RandomizedSearchCV(estimator=regressor_RFR, param_distributions=parameters, n_iter=50, cv=5, verbose=2, n_jobs=-1, random_state=0)

random_cv.fit(x_train, y_train)

random_cv.best_estimator_

random_cv.best_params_

#final model: random forest regressr
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)



















