import best_params
import gb_model
import pandas as pd
import make_predictions
import pom


names = ['AMD', 'SHAN', 'NVDA', 'AMZN', 'GOOGL', 'IBM']
dic_best_params = {}



def find_best_param_to_dic(names, dic_best_params): 
    for name in names:
        X_train, X_test, y_train, y_test = gb_model.open_csv(name)
        best_param_values = best_params.find_best_gbr_params(best_params.params, X_train, y_train, X_test, y_test)
        dic_best_params.update({f"{name}": best_param_values})
        
        print(f'{name} was added to dict success')
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


#find_best_param_to_dic(names, dic_best_params)
#print(dic_best_params)
#find_best_param_to_csv(names)
make_predictions.write_predicted_prices_from_csv(names)
#make_predictions.write_predicted_prices_from_dic(names, dic_best_params)

