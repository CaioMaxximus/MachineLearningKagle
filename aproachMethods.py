import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


label_target = 'Salary'
label_train = ['Number','Age','Weight']

my_imputer = SimpleImputer(strategy = 'most_frequent')
data = pd.read_csv('./nba.csv')
data.dropna(axis = 0, subset = [label_target], inplace = True)


def mae(x_train,x_valid, target_train,target_valid):
    model = RandomForestRegressor(random_state = 0)
    model.fit(x_train,target_train)
    prediction = model.predict(x_valid)
    error = mean_absolute_error(target_valid,prediction)
    print('ei')
    return error


def drop_rows():
    columns = label_train.copy()
    columns.append(label_target)
    local_data = data.dropna(axis = 0, subset = columns)

    #local_data = data.dropna(axis = 0, subset = ['Salary','Number','Age','Weight'])

    target = local_data[label_target]
    train = local_data[label_train]
 
    X_train , X_valid , target_train , target_valid = train_test_split(train, target, random_state = 0)

    print(("Drop rows method MAE >: %.1f" %(mae(X_train,X_valid,target_train,target_valid))))


def drop_columns():
    local_data = data
    target = local_data[label_target]
    train = local_data[label_train]

    X_train , X_valid , target_train , target_valid = train_test_split(train, target, random_state = 0)


    missing_values = [col for col in X_train.columns if X_train[col].isnull().any()] 
    reduced_X_train = X_train.drop(missing_values, axis = 1)
    reduced_X_valid = X_valid.drop(missing_values, axis = 1)

    print(("Drop colums  method MAE >: %.1f" %(mae(reduced_X_train, reduced_X_valid, target_train,target_valid))))



def imputation():

    local_data = data
    target = local_data[label_target]
    train = local_data[label_train]

    X_train , X_valid , target_train , target_valid = train_test_split(train, target, random_state = 0)

    imputed_x_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_x_valid = pd.DataFrame(my_imputer.transform(X_valid))

    imputed_x_train.columns = X_train.columns
    imputed_x_valid.columns = X_valid.columns

    

    print("Imputer  method MAE >: %.1f" %(mae(imputed_x_train, imputed_x_valid, target_train, target_valid)))

def imputation_extension():

    local_data = data
    y = data[label_target]
    x = data[label_train]
    
    X_train , X_valid ,target_train, target_valid = train_test_split(x,y,random_state = 0)

    col_with_misssing = [col for col in x.columns if x[col].isnull().any()]
    
    
    for col in col_with_misssing:
        X_train[col + 'missing'] = X_train[col].isnull()
        X_valid[col + 'missing'] = X_valid[col].isnull()
    
    imputed_x_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_x_valid = pd.DataFrame(my_imputer.transform(X_valid))
    
    imputed_x_train.columns = X_train.columns
    imputed_x_valid.columns = X_valid.columns
    
    print("Imputer extension MAE >: %1.f " %(mae(imputed_x_train,imputed_x_valid,target_train,target_valid)))

imputation()
drop_columns()
drop_rows()
imputation_extension()