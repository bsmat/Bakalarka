import numpy as np
import gb_model
import pandas as pd
import matplotlib.pyplot as plt


params = {'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2], 
            'tree_levels': [3, 4, 5, 6], 
            'number_trees': [100, 200, 500, 1000]
         }


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def find_best_gbr_params(params, X_train, y_train, X_test, y_test):
    best_params = None
    best_mse = float('inf')

    for lr in params['learning_rate']:
        for tl in params['tree_levels']:
            for nt in params['number_trees']:
                model = gb_model.GradientBoostingRegressorFromScratch(tl, nt, lr)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse_value = mse(y_test, y_pred)
                if mse_value < best_mse:
                    best_params = {'learning_rate': lr, 'tree_levels': tl, 'number_trees': nt}
                    best_mse = mse_value
    return best_params


def best_params_to_image():
    df = pd.read_csv('Data/best_params.csv')
    fig, ax = plt.subplots()
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    plt.show()

#best_param = find_best_gbr_params(params, gb_model.X_train, gb_model.y_train, gb_model.X_test, gb_model.y_test)
#print(best_param)
best_params_to_image()