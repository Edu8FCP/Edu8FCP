from os import supports_effective_ids
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import random as rand

samples = 40

plt.style.use('classic')
x = np.zeros(40)
y = np.zeros(40)

for i in range (10):
    x[i] = 1+7*i/10
    y[i] = 6 + 2*i/10

for i in range (10):
    x[i+10] = 8+i/10
    y[i+10] = 8 - 4*i/10

for i in range (10):
    x[i+20] = 9 - 7*i/10
    y[i+20] = 4 - 2*i/10

for i in range (10):
    x[i+30] = 2 - i/10
    y[i+30] = 2 + 4*i/10

print(x)
print(y)

#Plot
fig1, ax1 = plt.subplots()

ax1.scatter(x, y, color='g')

#plt.show()

plt.savefig("quadrado.png")