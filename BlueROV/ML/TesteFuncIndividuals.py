import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

def get_n_samples(x_data, y_data, n):
    indexes = np.round(np.linspace(0,99, n)).astype('int')
    return x_data[indexes], y_data[indexes]

#returns a  single random index from an array
def get_random_index(array_size):
        index = np.random.choice(array_size, 1)
        return index[0]
        #return 5
    
def build_dataset(x_,y_, shape):
        data = []
        row = {}
        for i in range(len(x_)):
            row['x' + str(i+1)] = x_[i]
            row['y' + str(i+1)] = y_[i]
        row['shape'] = shape  
        data.append(row)  
        return data  

def createCircle(radius, centre):
    theta = np.linspace(0, 2*math.pi,100)
    x_circle = radius * np.cos(theta) + centre[0]
    y_circle = radius * np.sin(theta) + centre[1]
    return x_circle, y_circle

def plotter(x_data, y_data, title):
    fig = plt.figure(figsize=[10,10]) 
    plt.plot(x_data,y_data,'b--')
    plt.xlabel('X-axis',fontsize=14)
    plt.ylabel('Y-axis',fontsize=14)
    plt.ylim(-18,18)
    plt.xlim(-18,18)
    plt.axhline(y=0, color ="k")
    plt.axvline(x=0, color ="k")
    plt.grid(True)
    saveFile = title + '.png' #o codigo de origem tinha .svg
    plt.savefig(saveFile)
    plt.show()

x, y = createCircle(2, centre = [0,0])
a = "TesteML_Circle"
plotter(x,y, a) 