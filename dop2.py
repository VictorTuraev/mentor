import numpy as np
import matplotlib.pyplot as plt
y = np.array([1., 2., 3., 4., 2., 1.])


n = len(y)//2
dt=0.001
b=1
m=0.1

c = np.array([1., -1, 1.])


def velocity(t, y):
	ans = np.zeros_like(y)
	for i in range(n):
		for j in range(n):
			if i !=j :
				ans[2*i] += (y[2*i+1]-y[2*j+1])/((y[2*i] - y[2*j])**2 + (y[2*i+1] - y[2*j+1])**2) 
				ans[2*i+1] += -(y[2*i]-y[2*j])/((y[2*i] - y[2*j])**2 + (y[2*i+1] - y[2*j+1])**2)
	return ans
 
"""def move(y, ans):
	for i in range(n):
		y[2*i+1] += ans[2*i+1]*dt 
		y[2*i] += ans[2*i]*dt """
def runge_kutta(y, t, dt, func):
	k1 = func(t, y)
	k2 = func(t + dt/2, y + dt/2*k1)
	k3 = func(t + dt/2, y + dt/2*k2)
	k4 = func(t + dt/2, y + dt*k3)
	return y + (dt/2)*(k1 + 2*k2 + 2*k3 + k4)


ans = []
for i in range(1000):
	y = runge_kutta(y, 1, dt, velocity)
	ans.append(y)


ans = np.transpose(np.array(ans))
plt.plot(ans[0], ans[1], 'r')
plt.plot(ans[2], ans[3], 'b')
plt.plot(ans[4], ans[5], 'g')
plt.show()
