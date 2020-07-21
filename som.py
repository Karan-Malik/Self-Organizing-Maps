
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Credit_Card_Applications.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
X=X[:,1:]
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
X=sc.fit_transform(X)

from minisom import MiniSom
som=MiniSom(x=10,y=10,input_len=14,sigma=1,learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X,num_iteration=100)

#Visualising results

from pylab import bone,pcolor,colorbar,show,plot
bone()
pcolor(som.distance_map().T) #used to get mean internodal dist for all
colorbar()
markers=['o','s']
colors=['r','g']
for i,x in enumerate(X):
    w=som.winner(x) #To get the winner node
    plot(w[0]+0.5,w[1]+0.5,markers[y[i]],markeredgecolor=colors[y[i]],markerfacecolor='None',markersize=10,markeredgewidth=2)
show()

#Identifying fraud customers
mappings=som.win_map(X)
frauds=(mappings[(6,4)])
frauds=sc.inverse_transform(frauds)