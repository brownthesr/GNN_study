import matplotlib.pyplot as plt
import numpy as np
w3 = 30
w2 = 100
L =12.4
w1 = np.linspace(0,500,500)
delta = L* np.cos(np.arctan(w1*w2/w3))
plt.plot(w1,delta)
plt.xlabel("w1")
plt.ylabel("delta")
plt.show()