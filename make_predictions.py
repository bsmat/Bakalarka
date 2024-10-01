import gb_model
import csv
import pandas as pd



def make_prediciton_from_dic(name_company, params_dict):
    params, metric = params_dict[name_company]
    learning_rate = params['learning_rate']
    tree_levels = params['tree_levels']
    number_trees = params['number_trees']
    #learning_rate = params_dict[name_company]['learning_rate']
    #tree_levels = params_dict[name_company]['tree_levels']
    #number_trees = params_dict[name_company]['number_trees']
    X_train, X_test, y_train, y_test = gb_model.open_csv(name_company)
    gb = gb_model.GradientBoostingRegressorFromScratch(tree_levels, number_trees, learning_rate)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    return y_pred

def make_predictions_from_csv(name_company):
    df = pd.read_csv('Data/best_params.csv')
    index = df.index[df['Company'] == f'{name_company}'].to_list()
    index = index[0]
    learning_rate = df.loc[index, 'Learning Rate']
    tree_levels = df.loc[index, 'Tree Levels']
    number_trees = df.loc[index, 'Number of Trees']
    X_train, X_test, y_train, y_test = gb_model.open_csv(name_company)
    X = gb_model.open_X(name_company)
    gb = gb_model.GradientBoostingRegressorFromScratch(tree_levels, number_trees, learning_rate)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X)
    return y_pred

def write_predicted_prices_from_csv(names_company):
    pred_prices = []
    for name in names_company:
        y_pred = make_predictions_from_csv(name)
        pred_prices.append(y_pred)

    with open('data/predicted_prices_csv.csv', 'w', newline='') as csvfile:
    # Создаем объект writer для записи CSV файла
        writer = csv.writer(csvfile)
    # Записываем заголовок таблицы
        writer.writerow(names_company)
    # Записываем данные из двух массивов
        for row in zip(*pred_prices):
            writer.writerow(row)

def write_predicted_prices_from_dic(names_company, params_dict):
    pred_prices = []
    for name in names_company:
        y_pred = make_prediciton_from_dic(name, params_dict)
        pred_prices.append(y_pred)


    with open('data/predicted_prices_dic.csv', 'w', newline='') as csvfile:
    # Создаем объект writer для записи CSV файла
        writer = csv.writer(csvfile)
    # Записываем заголовок таблицы
        writer.writerow(names_company)
    # Записываем данные из двух массивов
        for row in zip(*pred_prices):
            writer.writerow(row)

