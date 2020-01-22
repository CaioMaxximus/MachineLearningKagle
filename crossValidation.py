import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('./nba.csv')
data.dropna(axis = 0, inplace = True)

label_target = 'Salary'
label_train = ['Team','Weight','Position', 'Age','Height','College']

target = data[label_target] 
x = data[label_train]

categorical_columns = [col for col in x.columns if  x[col].dtype == 'object']
print(categorical_columns)
numerical_columns = [col for col in x.columns  if x[col].dtype in ['int64','float64']]
print(numerical_columns)
                     
final_columns = categorical_columns + numerical_columns


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

score = -1 * cross_val_score(my_pipeline, x, target,cv = 5 ,scoring='neg_mean_absolute_error')

print('mae >: %.1f', score.mean())