from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

#datas

data_frame = pd.read_csv("nba.csv")
data_frame = data_frame.dropna(axis = 0)
print(data_frame.describe())
x = data_frame[['Number','Age','Weight']]
y = data_frame['Salary']
#model 

train_x, val_x , train_y , val_y, = train_test_split(x,y,random_state = 0)

model = RandomForestRegressor(random_state= 1)
model.fit(train_x,train_y)
prediction = model.predict(val_x)
print("erro eh %.4f" %(mean_absolute_error(prediction,val_y)))