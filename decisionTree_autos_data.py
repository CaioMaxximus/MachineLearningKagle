from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import pandas as pd

oneh = OneHotEncoder()

def mae(max_leaf,train_x,train_y,val_x,val_y):
    #model = DecisionTreeRegressor(max_leaf_nodes = max_leaf, random_state= 1)
    model = oneh
    model.fit(train_x,train_y)
    prediction = model.predict(val_x)
    error = mean_absolute_error(prediction,val_y)
    return error

#getting datas

cars_datas = pd.read_csv("autos.csv")
cars_datas = cars_datas.dropna(axis = 0)
train_colums = ['yearOfRegistration','kilometer', 'vehicleType', 'gearbox','model','fuelType']
y = cars_datas.price
x = cars_datas[train_colums]
#x = oneh.fit_transform(x)

train_x , val_x , train_y, val_y = train_test_split(x, y, random_state = 0)
list = [20000]

for num in list:
    error = mae(num,train_x,train_y,val_x,val_y)
    print("leafs = %s ; error = %.3f " %(num ,error))  