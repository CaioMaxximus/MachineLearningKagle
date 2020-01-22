import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def mae(x_train,x_valid, target_train,target_valid):
    model = RandomForestRegressor(random_state = 0)
    model.fit(x_train,target_train)
    prediction = model.predict(x_valid)
    error = mean_absolute_error(target_valid,prediction)
    print('ei')
    return error




data = pd.read_csv('./nba.csv')
data.dropna(axis = 0, inplace = True)

label_target = 'Salary'
label_train = ['Team','Weight','Position', 'Age','Height','College']

target = data[label_target] 
x = data[label_train]

X_train_full, X_valid_full , target_train, target_valid = train_test_split(x, target, random_state = 0)

categorical_columns = [col for col in x.columns if  x[col].dtype == 'object']
print(categorical_columns)
numerical_columns = [col for col in x.columns  if x[col].dtype in ['int64','float64']]
print(numerical_columns)
                     
final_columns = categorical_columns + numerical_columns
X_train = X_train_full[final_columns].copy()
X_valid = X_valid_full[final_columns].copy()

numeric_transformer = SimpleImputer(strategy = 'constant')

categorical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
], verbose = True)

preprocessor = ColumnTransformer(transformers = [
    ('num', numeric_transformer, numerical_columns),
    ('cat', categorical_transformer,categorical_columns)
])

model = RandomForestRegressor(random_state = 0, n_estimators = 200)

my_pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('model', model)
])

my_pipeline.fit(X_train,target_train)

predction = my_pipeline.predict(X_valid)

mae = mean_absolute_error(target_valid, predction)
print(mae)