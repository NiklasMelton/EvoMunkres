import numpy as np
import matplotlib.pyplot as plt

mu = 0.7
sigma = 0.08
s = np.random.normal(mu, sigma, 1000)
s = np.maximum(np.minimum(s,200),-200)
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
plt.show()
