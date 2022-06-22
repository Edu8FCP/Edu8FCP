import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
#Create Database for these geometric curves
# https://github.com/cctech-labs/ml-2dshapes/blob/master/Conic_Shapes_Generator.ipynb

#------------------------------------------------------------------------

#Configurações Iniciais
#np.round - arredonda um array para um determinado nr de casas decimais
#.astype (converter para inteiro, mas para não truncar por baixo, antes arredonda)
#.linspace - é usado para criar uma sequencia uniformemente espaçados em um intervalo específico

#------------------------------------------------------------------------

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

#--------------------------------------------------------

#Gerador de funções
#Parabola, Circulo, Elipse, Hipérbola e rotação de coordenadas

#--------------------------------------------------------

def createRectangle(points):
    t = np.linspace(-5, 5, 100)
    x_rect =
    y_rect =
    

def createParabola(focal_length, centre, rotation):
    t = np.linspace(-math.pi, math.pi,100)
    x_parabola = focal_length * t**2
    y_parabola = 2 * focal_length * t
    if rotation is not None:
        x_parabola, y_parabola = rotateCoordinates(x_parabola, y_parabola, rotation) 
    x_parabola = x_parabola + centre[0]
    y_parabola = y_parabola + centre[1]
    return x_parabola, y_parabola

def createCircle(radius, centre):
    theta = np.linspace(0, 2*math.pi,100)
    x_circle = radius * np.cos(theta) + centre[0]
    y_circle = radius * np.sin(theta) + centre[1]
    return x_circle, y_circle

def createEllipse(major_axis, minor_axis, centre, rotation):
    theta = np.linspace(0, 2*math.pi,100)
    x_ellipse = major_axis * np.cos(theta) 
    y_ellipse = minor_axis * np.sin(theta) 
    if rotation is not None:
        x_ellipse, y_ellipse = rotateCoordinates(x_ellipse,y_ellipse, rotation)
    x_ellipse = x_ellipse + centre[0]
    y_ellipse = y_ellipse + centre[1]
    return x_ellipse, y_ellipse

def createHyperbola(major_axis, conjugate_axis, centre, rotation):
    theta = np.linspace(0, 2*math.pi,100)
    x_hyperbola = major_axis * 1/np.cos(theta) + centre[0]
    y_hyperbola = conjugate_axis * np.tan(theta) + centre[1]
    if rotation is not None:
        x_hyperbola, y_hyperbola = rotateCoordinates(x_hyperbola, y_hyperbola, rotation)
    x_hyperbola = x_hyperbola + centre[0]
    y_hyperbola = y_hyperbola + centre[1]
    return x_hyperbola, y_hyperbola

def rotateCoordinates(x_data, y_data, rot_angle):
    x_ = x_data*math.cos(rot_angle) - y_data*math.sin(rot_angle)
    y_ = x_data*math.sin(rot_angle) + y_data*math.cos(rot_angle)
    return x_,y_

#----------------------------------------------------------------------

#Código para ver o que as funções acima deram de output
#Plotter

#----------------------------------------------------------------------

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

#-----------------------------------------------------------------------
#
# "MAIN"
#
# Código para correr os programas
#
#-----------------------------------------------------------------------

#x,y = createParabola(focal_length= 1, centre= [10,10],rotation= math.pi/5)
#a = "TesteML_Circle"
#plotter(x,y, a)

#------------------------------------------------------------------------

#------------------------------------------------------------------------

#------------------------------------------------------------------------

#Simulação de Amostragem (Sampling)

#------------------------------------------------------------------------

sample_count = 6

#circle
x,y = createCircle(radius = 1, centre= [10,10])
get_n_samples(x,y, sample_count)

# Parabola
x,y = createParabola(focal_length= 1, centre= [10,10],rotation= math.pi/5)
get_n_samples(x,y, sample_count)

# Hyperbola
x,y = createHyperbola(major_axis= 2, conjugate_axis = 1, centre= [10,10],rotation= math.pi/5)
get_n_samples(x,y, sample_count)

# Ellipse
x,y = createEllipse(major_axis= 2, minor_axis= 1, centre= [10,10],rotation= math.pi/5)
get_n_samples(x,y, sample_count)


#------------------------------------------------------------------------

# Parabola
#Parâmetros a variar para criar vários exemplares
focal_length_array = np.linspace(1, 20, 100)
centre_x_arr = np.linspace(-12, 12, 100)
centre_y_arr = np.linspace(-12, 12, 100)
rotation_array = np.linspace(2*math.pi, 100)

#------------------------------------------------------------------------

#Biblioteca Pandas 
#DataFrame - é uma estrutura de dados bidimensional com colunas de potencialmente diferentes tipos

#Criação da base de dados. Adquire os parâmetros de forma aleatória. Cria a forma geométrica. Coloca na base

#------------------------------------------------------------------------

parabola_dataset = pd.DataFrame()

for i in range(1000):
    focal_length = focal_length_array[get_random_index(len(focal_length_array))]
    centre_x = centre_x_arr[get_random_index(len(centre_x_arr))]
    centre_y = centre_y_arr[get_random_index(len(centre_y_arr))]
    rotation = rotation_array[get_random_index(len(rotation_array))]
    x,y = createParabola(focal_length= focal_length, centre= [centre_x, centre_y],rotation= rotation)
    x_, y_ = get_n_samples(x, y, sample_count)
    data = build_dataset(x_, y_, 'parabola')
    parabola_dataset = parabola_dataset.append(data, ignore_index=True)

# Ellipse
major_axis_array = np.linspace(1,20,100)
minor_axis_array = np.linspace(1,20,100)
centre_x_arr = np.linspace(-12, 12, 100)
centre_y_arr = np.linspace(-12, 12, 100)
rotation_array = np.linspace(2*math.pi, 100)

ellipse_dataset = pd.DataFrame()

for i in range(1000):
    major_axis = major_axis_array[get_random_index(len(major_axis_array))]
    minor_axis = minor_axis_array[get_random_index(len(minor_axis_array))]
    centre_x = centre_x_arr[get_random_index(len(centre_x_arr))]
    centre_y = centre_y_arr[get_random_index(len(centre_y_arr))]
    rotation = rotation_array[get_random_index(len(rotation_array))]
    x,y = createEllipse(major_axis=major_axis, minor_axis=minor_axis, centre= [centre_x,centre_y], rotation= rotation)
    x_,y_ = get_n_samples(x, y, sample_count)
    data = build_dataset(x_, y_, 'ellipse')
    ellipse_dataset = ellipse_dataset.append(data, ignore_index=True)  

# Hyperbola
major_axis_array = np.linspace(1,20,100)
conjugate_axis_array = np.linspace(1,20,100)
centre_x_arr = np.linspace(-12, 12, 100)
centre_y_arr = np.linspace(-12, 12, 100)
rotation_array = np.linspace(2*math.pi, 100)

hyperbola_dataset = pd.DataFrame()

for i in range(1000):
    major_axis = major_axis_array[get_random_index(len(major_axis_array))]
    conjugate_axis = conjugate_axis_array[get_random_index(len(conjugate_axis_array))]
    centre_x = centre_x_arr[get_random_index(len(centre_x_arr))]
    centre_y = centre_y_arr[get_random_index(len(centre_y_arr))]
    rotation = rotation_array[get_random_index(len(rotation_array))]
    x,y = createHyperbola(major_axis=major_axis, conjugate_axis=conjugate_axis, centre= [centre_x,centre_y], rotation= rotation)
    x_,y_ = get_n_samples(x, y, sample_count)
    data = build_dataset(x_, y_, 'hyperbola')
    hyperbola_dataset = hyperbola_dataset.append(data, ignore_index=True)   

# Circle
radius_array = np.linspace(1,20,100)
centre_x_arr = np.linspace(-12, 12, 100)
centre_y_arr = np.linspace(-12, 12, 100)

circle_dataset = pd.DataFrame()

for i in range(1000):
    radius = radius_array[get_random_index(len(radius_array))]
    centre_x = centre_x_arr[get_random_index(len(centre_x_arr))]
    centre_y = centre_y_arr[get_random_index(len(centre_y_arr))]
    x,y = createCircle(radius = radius, centre= [centre_x,centre_y])
    x_,y_ = get_n_samples(x, y, sample_count)
    data = build_dataset(x_, y_, 'circle')
    circle_dataset = circle_dataset.append(data, ignore_index=True)

#-------------------------------------------------------------------
# 
# Concatena as várias formas criadas e combina tudo numa dataset
# Cria a dataset
#
# ------------------------------------------------------------------   
combined_dataset = pd.concat([parabola_dataset, ellipse_dataset, hyperbola_dataset, circle_dataset])
combined_dataset.to_csv('Conic-Section_dataset.csv', index=False)

#--------------------------------------------------------------
# STEP 3 - TRAINING

#Vamos treinar uma rede neutral
#Vamos usar TensorFlow em conjunto com Keras para repidamente desenhar e treinar

#--------------------------------------------------------------

"""
data = pd.read_csv('Conic-Section_dataset.csv', index_col=False)
data= data.sample(frac=1, random_state=42).reset_index()
data.drop(['index'], 1, inplace=True)

#Vamos converter as etiquetas em valores inteiros para ter significado 

X = data.values[:,1:]
Y = data.values[:,0]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

Y : ["parabola" , "ellipse" , "circle" , "hyperbola"]
encoded_Y : [0 , 1 , 2 , 3]
dummy_Y : [1, 0, 0, 0] , [0, 1, 0, 0] , [0, 0, 1, 0] , [0, 0, 0, 1]

#Agora estamos prontos para criar a nossa rede neural usando o Keras

"""
#Let’s start off by defining the function that creates our model. 
#To create a neural network, we will use the Sequential API of keras. 
#We add four fully connected hidden layers with [128, 256, 64, 32] neurons respectively in each layer. 
#After each hidden layer, we use the ReLu activation unit for adding non-linearity to the output of each layer. 
#This helps ensure that our neural network model can learn complex patterns, and also gives a continuous, non-negative output.

#Neural network training generally tend to overfit the input, which means the model could get unnecessarily complex. 
#To control the complexity, we also add a dropout percentage after each layer. 
#The final layer has a softmax function to produce output.
"""
STEP 3: Training a Machine Learning Algorithm
Now comes the really interesting part - training a machine learning algorithm!
We will train a neural network classifier for our problem. We choose neural networks as they are capable of learning very complex functions. We will use Tensorflow along with Keras to quickly and simply design and train neural networks.
Let’s create a new notebook and import the dataset we generated earlier. The following code snippet demonstrates how to read our dataset csv file using pandas:
data = pd.read_csv('Conic-Section_dataset.csv', index_col=False)
data= data.sample(frac=1, random_state=42).reset_index()
data.drop(['index'], 1, inplace=True)
ML Training dataset for conic shape
Let’s process out data first by converting the string labels ('circle' / 'parabola' / 'ellipse' / 'hyperbola') into integer values. Why do we need to do this? It's because the machine learning algorithm is mathematical, and strings don’t mean anything to it!
We will convert our labels into integer values. Since we have four classes, we will get four integer values (0,1,2,3). To convert to integer values, we use the LabelEncoder class from scikit-learn.
X = data.values[:,1:]
Y = data.values[:,0]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
Then we convert it into "one-hot encoding". Basically, this step converts:
Y : [“parabola” , “ellipse” , “circle” , “hyperbola”]
Into this
encoded_Y : [0 , 1 , 2 , 3]
Which then becomes
dummy_Y : [1, 0, 0, 0] , [0, 1, 0, 0] , [0, 0, 1, 0] , [0, 0, 0, 1]
These are our labels, in “one-hot-encoded” form. We will use dummy_Y as label for our algorithm.
We are now ready to create our neural network model using Keras.
Let’s start off by defining the function that creates our model. To create a neural network, we will use the Sequential API of keras. We add four fully connected hidden layers with [128, 256, 64, 32] neurons respectively in each layer. After each hidden layer, we use the ReLu activation unit for adding non-linearity to the output of each layer. This helps ensure that our neural network model can learn complex patterns, and also gives a continuous, non-negative output.
Neural network training generally tend to overfit the input, which means the model could get unnecessarily complex. To control the complexity, we also add a dropout percentage after each layer. The final layer has a softmax function to produce output.
def baseline_model():
# create model
    model = Sequential()
    model.add(Dense(128, input_dim=12, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
# Compile model
    Adadelta =  optimizers.Adadelta(lr = 1)
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta, metrics=['accuracy'])
    return model

"""
#Finally, we are using the categorical cross entropy for computing the loss during training and an efficient optimization algorithm Adadelta for training the data set.
#After compiling the model, we call the model.fit() function to start the training on our data. 
#In the model.fit(), we specify the features and targets. 
#We also specify that the algorithm should use 80% data for training and 20% data as unseen data(validation_split=0.2), for performing inference.

#"""
#history = model.fit(x=X,y=dummy_y,validation_split=0.2,shuffle=True, epochs=200, batch_size=12)