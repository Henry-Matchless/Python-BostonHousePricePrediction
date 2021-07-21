import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x=[]
y=[]

### Generate data, using inputs (x-values) between 1 and 200 in increments of 2 ###
for i in range(1,200,2):
    x.append(i)
### use random fluctuations: y = 0.4*x + 3 +, drawn from a uniform distribution between -10 and +10 ###
for i in x:
    RAND=i*0.4+3+np.random.uniform(low=-10,high=10)
    y.append(RAND)

### reshape and divide the training dataset and testing dataset into a 7:3 ratio for using ###
data_x=np.array(x).reshape(-1,1)
data_y=np.array(y).reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 The Activation function used in module A is Logistic, the number of neurons is 1, and the number of iterations is 1000
 The Activation function used in module B is Logistic, the number of neurons is 1, and the number of iterations is 10000
 The Activation function used in module C is Logistic, the number of neurons is 40, and the number of iterations is 10000
 The Activation function used in module D is Relu, the number of neurons is 1, and the number of iterations is 10000
 
 By comparing module A with module B, we can see the influence of the number of iterations on the result in the same Activation function and the same number of neurons
 By comparing module B with module C, we can see the influence of the number of neurons in the same Activation function and the same iteration number on the result
 By comparing Module B with Module D, we can see the influence of different Activation functions on the effect under the same number of iterations and the same number of neurons 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"A"
clf_logic1 = MLPRegressor(hidden_layer_sizes=(1), max_iter=1000,activation='logistic')
plt.title('Activation function: logistic, hidden_layer_sizes: 1, iteration: 100',fontsize=10)
clf_logic1.fit(x_train, y_train)
y_predict = clf_logic1.predict(x_test)
plt.scatter(x_test,y_test)
plt.plot(x_test,y_predict,c="blue")
plt.show()

"B"
clf_logic2 = MLPRegressor(hidden_layer_sizes=(1), max_iter=10000,activation='logistic') #regressor
### title ###
plt.title('Activation function: logistic, hidden_layer_sizes: 1, iteration: 10000 ',fontsize=10)
clf_logic2.fit(x_train, y_train)
y_predict = clf_logic2.predict(x_test) #prediction
plt.scatter(x_test,y_test)
plt.plot(x_test,y_predict,c="green")
plt.show()

"C"
clf_logic3 = MLPRegressor(hidden_layer_sizes=(40), random_state=1, max_iter=10000,activation='logistic')
plt.title('Activation function: logistic, hidden_layer_sizes: 40, iteration: 10000 ',fontsize=10)
clf_logic3.fit(x_train, y_train)
y_predict = clf_logic3.predict(x_test)
plt.scatter(x_test,y_test)
plt.plot(x_test,y_predict,c="magenta")
plt.show()

"D"
clf_relu = MLPRegressor(hidden_layer_sizes=(1), max_iter=10000,activation='relu')
plt.title('Activation function: relu, hidden_layer_sizes: 1, iteration: 10000 ',fontsize=10)
clf_relu.fit(x_train, y_train)
y_predict = clf_relu.predict(x_test)
plt.scatter(x_test,y_test)
plt.plot(x_test,y_predict,c="brown")
plt.show()

