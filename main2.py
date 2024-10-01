import best_params 
import compare_models
import gb_model
import make_predictions
import pom
import random_forest
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor



names = ['AMD', 'SHAN', 'NVDA', 'AMZN', 'GOOGL', 'IBM']
existing_libs = [GradientBoostingRegressor, RandomForestRegressor]
dic_best_params = {}
df = pd.read_csv('Data/predicted_prices_csv.csv')
returns = pom.pct_change(df)
mean_returns = pom.mean(returns)
cov_matrix = pom.covariance_matrix(returns)
num_portfolios = 20000
risk_free_rate = 0.0178

def find_best_param_to_dic(names, dic_best_params): 
    for name in names:
        X_train, X_test, y_train, y_test = gb_model.open_csv(name)
        best_param_values = best_params.find_best_gbr_params(best_params.params, X_train, y_train, X_test, y_test)
        dic_best_params.update({f"{name}": best_param_values})
        
        print(f'{name} was added to dict success')
    print()
    return dic_best_params




def find_best_param_to_csv(names):
    mse = []
    learning_rates = []
    tree_levels = []
    number_trees = []   
    for name in names:
        X_train, X_test, y_train, y_test = gb_model.open_csv(name)
        best_param_values = best_params.find_best_gbr_params(best_params.params, X_train, y_train, X_test, y_test)
        data_turple = best_param_values[0]
        learning_rates.append(data_turple['learning_rate'])
        tree_levels.append(data_turple['tree_levels'])
        number_trees.append(data_turple['number_trees'])
        mse.append(best_param_values[1])
    
    df = pd.DataFrame({
    'Company': names,
    'Learning Rate': learning_rates,
    'Tree Levels': tree_levels,
    'Number of Trees': number_trees,
    'MSE': mse
    })

    df.to_csv('Data/best_params.csv', index=False)




print('select commands from the list: 1 - find best parameters for grafient boosting, 2 - compare models, 3 - make predictions, 4 - build the efficient portfolio')
com = int(input())
match com:
    case 1:
        find_best_param_to_csv(names)
        print('best paramaters for gradient boosting model was added to csv file best_params.csv')
    case 2:
        find_best_param_to_dic()
        print('best parameters for gradient boosting model was added to dictionary')
    case 3:
        compare_models.compare_libs(existing_libs, names)
        print('comparison table is saved as comparison_table.png file')
    case 4:
        make_predictions.write_predicted_prices_from_csv(names)
        print('predicted prices are saved as predicted_prices.csv')
    case 5:
        pom.show_efficient_values(mean_returns, cov_matrix, risk_free_rate)
        print('below the efficient values is presented')
    case 6:
        pom.show_efficient_graphic(mean_returns, cov_matrix, risk_free_rate)
        print('the efficient frontier is saved as efficient_frontier.png')
    case 7:
        pom.save_val_to_image(mean_returns, 'mean_returns')
        print('mean returns for assets is saved in mean_returns.png file')
    case 8:
        pom.save_val_to_image(cov_matrix, 'cov_matrix')
        print('covariance matrix is saved in cov_matix.png file')
    case 9:
        print(returns)
        pom.save_val_to_image(returns, 'returns')
        print('the price changes is presented, also save as returns.png file')
    case _:
        print('the wrong command was selected')
        print('select commands from the list')


print('')