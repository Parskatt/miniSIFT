import numpy as np
import matplotlib.pyplot as plt

def cov(x,l=np.pi/4):
    K =np.exp(-2*np.sin(np.abs(x-x.T)/2)**2/l**2)
    return K


x = np.linspace(0,2*np.pi)[np.newaxis,:]
K = cov(x)
f = np.sin(x)

plt.imshow(K)
plt.show()