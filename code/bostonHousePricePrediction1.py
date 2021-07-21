from sklearn.datasets import load_boston
import sklearn.decomposition as sk_decomposition
import sklearn.preprocessing as sk_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import r2_score#R square
from sklearn import ensemble
import numpy as np
### load the data of boston_dataset ###
X, y = load_boston(return_X_y=True)

### the size of X is 506*13 ###
print(X.shape)
### the size of Y is 506*1 ###
print(y.shape)
### field ’DESCR’ -- it contains important information about the dataset, ###
print(load_boston().DESCR)

### I used "StandardScaler" method to preprocess the data ###
scaler = sk_preprocessing.StandardScaler().fit(X)
data_x = scaler.transform(X) #transform

### Divide the training dataset and testing dataset into a 7:3 ratio for using ###
x_train, x_test, y_train, y_test = train_test_split(data_x,y, test_size=0.3)

### The model for regression I used is random forest algorithm ###
### The larger the N_ESTIMATORS, the better, but the memory and training and prediction time will increase accordingly,
### and the marginal benefit will be diminishing. Therefore,
### N_ESTIMATORS should be selected as large as possible within the affordable memory/time，so I choose 20 for it ###
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)
model_RandomForestRegressor.fit(x_train, y_train)
y_predict = model_RandomForestRegressor.predict(x_test) # prediction
# print("predict results of the Boston house price for x_test data are:\n",y_predict)

### evaluation: RMSE and MAE ###
print("RMSE:",np.sqrt(mean_squared_error(y_test,y_predict)))
print("MAE:",mean_absolute_error(y_test,y_predict))

### data processing for testingset.npy ###
test_npy=np.load("testingset.npy")
mean = np.mean(X, axis=0)
standard_deviation = np.std(X, axis=0)
"""""
z = (x - u) / s
"""""
data_x_for_pred = np.divide(np.subtract(test_npy, mean), standard_deviation)

### making prediction of the data of testingset ###
y_predict = model_RandomForestRegressor.predict(data_x_for_pred)
print("predict results of the Boston house price are:\n",y_predict)
