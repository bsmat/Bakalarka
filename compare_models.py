import gb_model
import random_forest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



existing_libs = [GradientBoostingRegressor, RandomForestRegressor]
names = ['AMD', 'AMZN', 'GOOGL', 'IBM', 'NVDA', 'SHAN']
np.random.seed(42)
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def make_model_from_existing(libnames, filenames):
    res = {}
    for libname in libnames:
        list_of_mse = []
        for filename in filenames:
            df = pd.read_csv(f'Data/{filename}.csv')
            X=df[['Open', 'High', 'Low', 'Close', 'Volume']].values
            y=df['Adj Close'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
            model = libname()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            list_of_mse.append(mse(y_test, y_pred))
        score = (np.sum(list_of_mse)) / len(list_of_mse)
        class_name = libname.__name__
        res[f'{class_name}'] = score

    list_of_mse = []
    for filename in filenames:
        df = pd.read_csv(f'Data/{filename}.csv')
        X=df[['Open', 'High', 'Low', 'Close', 'Volume']].values
        y=df['Adj Close'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        model = xg.XGBRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        list_of_mse.append(mse(y_test, y_pred))
    score = (np.sum(list_of_mse))/len(list_of_mse)
    res['XGBoosting'] = score

    return res

def make_model_from_mylibs(company_names):
    res = {}
    list_of_mse =[]
    param = pd.read_csv('Data/best_params.csv')
    for company_name in company_names:
        index = param.index[param['Company'] == f'{company_name}'].to_list()
        index = index[0]
        MSE = param.loc[index, 'MSE']
        list_of_mse.append(MSE)
    score = (np.sum(list_of_mse))/len(list_of_mse)
    res['GB'] = score

    list_of_mse = []
    for company_name in company_names:
        df = pd.read_csv(f'Data/{company_name}.csv')
        X=df[['Open', 'High', 'Low', 'Close', 'Volume']].values
        y=df['Adj Close'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        rf = random_forest.RandomForestRegressor(n_estimators=10, max_depth=5)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        list_of_mse.append(mse(y_test, y_pred))    
    score = (np.sum(list_of_mse))/len(list_of_mse)
    res['RF'] = score

    return res

def compare_libs(existing_libs, names):
    res_dic = {**make_model_from_existing(existing_libs, names),**make_model_from_mylibs(names) }

    df = pd.DataFrame(list(res_dic.items()), columns=['Model', 'MSE'])
    plt.figure(figsize=(8, 4))
    table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.05, 1.2)
    plt.axis('off')
    plt.savefig('Images/comparison_table.png')  
    plt.show()


compare_libs(existing_libs, names)
