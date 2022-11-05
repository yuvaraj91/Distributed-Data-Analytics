import numpy as np
import matplotlib.pyplot as plt

workers = [1, 2, 4, 6, 8]
Tp = [0.52953505, 0.28256410, 0.25546333, 0.16265410, 0.13409640]
Ts = Tp[np.argmin(Tp)]
Sp = [Ts / i for i in Tp]
linear = np.linspace(0, Tp[np.argmax(Tp)], 5, endpoint=True)

plt.plot(workers, Sp, color='red')
plt.plot(workers, linear, color='black')
plt.title('Parallel Speedup & Efficiency')
plt.xlabel('Number of workers, P')
plt.ylabel('Sp')
plt.show()
