import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,LassoCV,RidgeCV,ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures  #多项式特征
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
import warnings

# ignore warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)

# Pipeline for grid search
models = [
    Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('linear', RidgeCV(alphas=np.logspace(-3,1,20)))
        ]),
    Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('linear', LassoCV(alphas=np.logspace(-3,1,20))) #logspace 以10为底，从10的-3次方止10的0次方，中间有20步
        ]),
    Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('linear', LinearRegression())
        ]),
    Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('linear', ElasticNetCV(alphas=np.logspace(-3,1,20)))
        ])
]

# parameters dict
parameters = {
    "poly__degree": [3, 2, 1],
    "poly__interaction_only": [True, False],
    "poly__include_bias": [True, False],
    "linear__fit_intercept": [True, False]
}

titles = ['Ridge', 'Lasso', 'LinearRegression', 'ElasticNet']
colors = ['g-', 'b-', 'y-', 'c-']


# load data
def load_data_set():
    boston = datasets.load_boston()
    print(boston.DESCR)

    X = boston.data
    y = boston.target
    return X, y


# train model
def train(X_train, y_train):
    model_reg = []
    for i in range(len(models)):
        model = GridSearchCV(models[i], param_grid=parameters, cv=5, n_jobs=1)
        model.fit(X_train, y_train)
        model_reg.append(model)
    return model_reg

# test data
def test_model(model_reg, X_test, y_test):
    model_r2_score = []
    model_mse = []

    for i, model in enumerate(model_reg):
        y_hat = model.predict(X_test)
        model_r2_score.append(r2_score(y_test, y_hat))
        model_mse.append(metrics.mean_squared_error(y_test, y_hat))
        print('%s：best parameters：' % titles[i], model.best_params_)
        print('%s：r2_score=%.3f' % (titles[i], r2_score(y_test, y_hat)))
        print('%s：MSE=%.3f' % (titles[i], metrics.mean_squared_error(y_test, y_hat)))

    return model_r2_score, model_mse


# plot predict result
def plot_result(X_test, y_test, model_reg, model_mse):
    plt.figure(figsize=(16, 8), facecolor='w')
    ln_x_test = range(len(X_test))
    plt.plot(ln_x_test, y_test, 'r-', lw=2, label='actual')

    for i, model in enumerate(model_reg):
        y_predict = model.predict(X_test)
        plt.plot(ln_x_test, y_predict, colors[i], lw=2,
                 label='%s: MSE=%.3f' % (titles[i], model_mse[i]))

    plt.legend(loc='lower right')
    plt.grid(True)
    plt.title('Predict price')
    plt.savefig('res.png')
    plt.show()

# select model by mse
def select_model(model_mse, model_reg):
    min_mse_index = model_mse.index(min(model_mse))
    print("%s is selected!" % titles[min_mse_index])
    return model_reg[min_mse_index]


# predict price
def predict_price(lin_reg):
    print("predict price is >>>>>>>>>>>>>>")
    data =np.load('testingset.npy')
    print(lin_reg.predict(data))





def main():
    # get data
    X, y = load_data_set()
    # get train data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # training data
    model_reg = train(X_train, y_train)
    # test data
    model_r2_score, model_mse = test_model(model_reg, X_test, y_test)

    # plot predict result
    plot_result(X_test, y_test, model_reg, model_mse)

    # select model
    model = select_model(model_mse, model_reg)

    # predict price
    predict_price(model)



if __name__ == '__main__':
    main()

