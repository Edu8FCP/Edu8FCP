from os import supports_effective_ids
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import turtle

plt.style.use('classic')
"""x = np.zeros(8)
y = np.zeros(8)

#2 método
#for i in range(8):
 #   x[i] = float(rand.randint(0,25))
  #  y[i] = float(rand.randint(0,25))
#1 método    
#x = np.array([0.1, 1.1, 3.1, 4.1, 6.1, 8.1, 9.1, 14.1, 19.1, 22.1])
#y = [4, 6, 7, 7, 9, 11, 14, 17, 23, 26]

# NEW Bubble sort para ordenar
n = len(x)
 
# Atravessar todos os elementos
for i in range(n-1):
# range(n) also work but outer loop will
# repeat one time more than needed.
 
    # Last i elements are already in place
    for j in range(0, n-i-1):
 
        # traverse the array from 0 to n-i-1
        # Swap if the element found is greater
        # than the next element
        if x[j] > x[j + 1] :
            x[j], x[j + 1] = x[j + 1], x[j]    

x_mean = 0
y_mean = 0

for i in range (len(x)):
    x_mean += x[i]
    y_mean += y[i]

soma1=0
soma2=0

#calculo forma 1
for i in range (len(x)):
    soma1 += (x[i]-x_mean)*(y[i]-y_mean)
    soma2 += pow(x[i]-x_mean, 2)

media = soma1/soma2    
media = float(media)
b = y_mean - media*x_mean

#função para gerar a melhor fit - forma 2
m, c = np.polyfit(x, y, 1)
m = float(m)

fig1, ax1 = plt.subplots()
#fig2, ax2 = plt.subplots()
"""

for i in range(2):
    turtle.forward(20)
    turtle.right(90)
    turtle.forwart(10)

#Plot
ax1.scatter(x, y, color='g')
ax1.plot(x,media*x+b, color='b')
ax1.plot(x, m*x+c, color='red')

plt.show()

plt.savefig("square1.png")
