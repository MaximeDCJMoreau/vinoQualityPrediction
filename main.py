from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.model_selection import train_test_split
from torch import nn
import numpy as np
import pandas as pd

# Hyperparameters parameters:

P = {
'epochs'         : 5,        # Number of epochs
'batch_size'     : 50,        # Number of propagation evaluated before updating
'layers'         : [12, 50, 50,  1],# Input - Hidden layers - Output
'l_r'            : 1e-3,      # Learning rate, 
'lambda_L2'      : 1e-4,      # L2 regularizer 
}

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_ (0.01)
# fetch dataset 

if __name__ == '__main__':
    wine_quality = fetch_ucirepo(id=186) 

    # data (as pandas dataframes) 
    Inputs = wine_quality.data.features 
    targets = wine_quality.data.targets 

    # Plot distribution of data

    fig, axs = plt.subplots(4, 3, figsize=(7, 7))
    for i, columnsName in enumerate(Inputs.columns):
        sns.histplot(pd.DataFrame((Inputs.loc[:,columnsName])), x = columnsName, kde = True,bins =12, ax=axs[i%4,i//4])
    
plt.show()

  