import numpy as np
import matplotlib.pyplot as plt
x = np.arange(5)
y = x
plt.plot(x, y, '-o')
plt.savefig('./output/result.png')
plt.show()
