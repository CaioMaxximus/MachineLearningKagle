import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
label_enconder  = LabelEncoder()
hot_econder = OneHotEncoder(handle_unknown = 'ignore' , sparse = False)

data = pd.read_csv('./nba.csv')
data.dropna(axis = 0, inplace = True)

label_target = 'Salary'
label_train = ['Team','Weight','Position', 'Age','Height','College']

target = data[label_target] 
x = data[label_train]

s = (x.dtypes == 'object')
categorical_columns = list(s[s].index)

def mae(X_train, X_valid, target_train, target_valid):
    model = RandomForestRegressor(random_state=0)
    model.fit(X_train,target_train)
    prediction = model.predict(X_valid)
    return mean_absolute_error(prediction,target_valid)



def label_enconder_():

    
    X_train , X_valid, target_train, target_valid = train_test_split( x,target ,random_state = 0)
    
    label_X_train = X_train.copy()
    label_X_valid = X_valid.copy()
    
    
    for col in categorical_columns:
        label_enconder.fit(x[col])
        print(col)
        label_X_train[col] = label_enconder.transform(X_train[col])
        label_X_valid[col] = label_enconder.transform(X_valid[col])
        
    print(label_X_train.head())
    print(label_X_valid.head())
        
    print("LabelEncoder mae >: %1.f" %(mae(label_X_train,label_X_valid,target_train,target_valid)))    
    
    
def one_hot_encoder():
    
    X_train, X_valid, target_train, target_valid = train_test_split(x, target, random_state = 0)
    
    train_hot = pd.DataFrame(hot_econder.fit_transform(X_train[categorical_columns]))
    valid_hot = pd.DataFrame(hot_econder.transform(X_valid[categorical_columns]))
    
    train_hot.index = X_train.index
    valid_hot.index = X_valid.index
    
    # X_train.drop(c ategorical_columns, axis = 1 , inplace = True)
    # X_valid.drop(categorical_columns, axis = 1, inplace =True)

    num_X_train = X_train.drop(categorical_columns, axis = 1) 
    num_X_valid = X_valid.drop(categorical_columns, axis = 1)
    
    # final_train_hot = pd.concat([X_train, train_hot], axis = 1)
    # final_valid_hot = pd.concat([X_valid , valid_hot], axis = 1)
    
    final_train_hot = pd.concat([num_X_train, train_hot], axis = 1)
    final_valid_hot = pd.concat([num_X_valid , valid_hot], axis = 1)
    
    print("OneHot mae>: %.1f" %(mae(final_train_hot,final_valid_hot, target_train, target_valid)))
    
label_enconder_()
one_hot_encoder()