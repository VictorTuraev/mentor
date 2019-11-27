import numpy as np
import matplotlib.pyplot as plt

gamma = 0.1
omega = 1.
dt = 0.002
y = np.array([1., 2., 3., 4., 0., 0., 0., 0.])

b=1
c = np.array([1., -1])
m=0.1


def f(t, y):
	return np.array([y[1], - 2*gamma*y[1] - omega**2 * y[0]])

def velocity(t, y):
	n = len(y) // 4
	z = y.reshape((2 * n, 2))
	ans = np.zeros_like(y)
	ans[0:2*n] = y[2*n:]
	for i in range(n):
		ans[2*(n+i)] += y[2*(n+i)+1]*b*c[i]/(-m)
		ans[2*(n+i)+1] -= y[2*(n+i)]*b*c[i]/(-m)
		#Fx=c[i]*c[j]*(x[i]-x[j])/((x[i]-x[j])**2+(y[i]-y[j])**2)
		r_cur = y[2*i:2*(i+1)]
		for j in range(n):
			if i != j:
				r_new = y[2*j:2*(j+1)]
				ans[2*(n+i):2*(n+i+1)] += c[i]*c[j]*(r_cur - r_new)/(np.linalg.norm(r_cur - r_new)**2)/m
	return ans

def runge_kutta(y, t, dt, func):
	k1 = func(t, y)
	k2 = func(t + dt/2, y + dt/2*k1)
	k3 = func(t + dt/2, y + dt/2*k2)
	k4 = func(t + dt/2, y + dt*k3)
	return y + dt/2*(k1 + 2*k2 + 2*k3 + k4)

ans = []
for i in range(1000):
	y = runge_kutta(y, 1, dt, velocity)
	ans.append(y[:4])



ans = np.transpose(np.array(ans))
plt.plot(ans[0], ans[1], 'r')
plt.plot(ans[2], ans[3], 'b')
plt.show()
