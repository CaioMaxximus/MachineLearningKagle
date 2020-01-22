import pandas as pd
import sys
from sklearn.tree import DecisionTreeRegressor as tree_R
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def mae(max_leaf, val_x, val_y, train_x, train_y):
    new_model = tree_R(max_leaf_nodes= max_leaf , random_state= 1)
    new_model.fit(train_x,train_y)
    prediction = new_model.predict(val_x)
    return mean_absolute_error(prediction, val_y)

    
def main():
    le = OneHotEncoder()
    #Fonte 
    nba_data = pd.read_csv("C:/Users/caios/Documents/ProjetosPessoaisLinguagens/MachineLearning_Iniciante/nba.csv")
    nba_data = nba_data.dropna(axis = 0)
    # Variaveis do treinamento
    predict_y = nba_data['Salary']
    data_frame_x = nba_data[['Number','Age','Weight']]
    data_frame_x = le.fit_transform(data_frame_x)
    print(list(data_frame_x))
    train_x , val_x , train_y , val_y =  train_test_split(data_frame_x, predict_y, random_state = 0) 

    leafs_size = list(range(25,35))
    menor = sys.maxsize
    max_leaf = leafs_size[0]
    for i in leafs_size:
        atual = mae(i,val_x, val_y, train_x , train_y)
        print(i, "Erro da previsao: %.2f" %(atual), sep= ' ')
        if(atual < menor):
            menor = atual
            max_leaf = i
    print(max_leaf)
main()
