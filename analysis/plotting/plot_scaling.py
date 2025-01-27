import matplotlib.pyplot as plt
import numpy as np

# Strong

time = np.array([5.559008161226909, 3.3812309225400288,2.7561391909917194, 2.3545579950014752 , 2.198979167143504])
speed = time/time[0]
proc = np.array([1,2,3,4,5])



x = np.linspace(1,5,100)
y = x
plt.plot(x,y,'r')
plt.plot(proc,speed,'b')
plt.xlabel("Number Processors")
plt.ylabel("Speedup")
plt.legend(['ideal scaling', 'real scaling'])
plt.title("Strong Scaling")
plt.show()

import pdb; pdb.set_trace()
plt.close()
# Weak

time = np.array([])
proc = np.array([1,3,5])
speed = time/proc
x = np.linspace(1,5,100)
y = x/x
plt.plot(x,y,'r')
plt.plot(proc,speed,'b')
plt.xlabel("Number Processors")
plt.ylabel("Time Per Processor")
plt.legend(['ideal scaling', 'real scaling'])
plt.title("Weak Scaling")
plt.show()
