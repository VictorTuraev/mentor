import numpy as np
import matplotlib.pyplot as plt
y = np.array([1., 2., 3., 4., 2., 1., 3., 2.])
n = len(y) // 2
dt = 0.01
c = np.array([1., -1., 1., -1.])
b = 1.
m = 0.01

def velocity1(t, y):
	ans = np.zeros_like(y)
	for i in range(n):
		for j in range(n):
			if i !=j :
				ans[2*i] += -c[j]*(y[2*i+1]-y[2*j+1])/((y[2*i] - y[2*j])**2 + (y[2*i+1] - y[2*j+1])**2) 
				ans[2*i+1] += c[j]*(y[2*i]-y[2*j])/((y[2*i] - y[2*j])**2 + (y[2*i+1] - y[2*j+1])**2)
	return ans
 

def runge_kutta(y, t, dt, func):
	k1 = func(t, y)
	k2 = func(t + dt/2, y + dt/2*k1)
	k3 = func(t + dt/2, y + dt/2*k2)
	k4 = func(t + dt/2, y + dt*k3)
	return y + (dt/2)*(k1 + 2*k2 + 2*k3 + k4)


def f(t, y):
	return np.array([y[1], - 2*gamma*y[1] - omega**2 * y[0]])

def velocity2(t, y):
	n = len(y) // 4
	z = y.reshape((2 * n, 2))
	ans = np.zeros_like(y)
	ans[0:2*n] = y[2*n:]
	for i in range(n):
		ans[2*(n+i)] += y[2*(n+i)+1]*b*c[i]/(-m)
		ans[2*(n+i)+1] -= y[2*(n+i)]*b*c[i]/(-m)
		r_cur = y[2*i:2*(i+1)]
		for j in range(n):
			if i != j:
				r_new = y[2*j:2*(j+1)]
				ans[2*(n+i):2*(n+i+1)] += c[i]*c[j]*(r_cur - r_new)/(np.linalg.norm(r_cur - r_new)**2)/m
	return ans


N = 100
M = 100000

ans = []
for i in range(N):
	y = runge_kutta(y, 1, dt, velocity1)
	ans.append(y)

ans = np.transpose(np.array(ans))
plt.plot(ans[0], ans[1], 'r')
plt.plot(ans[2], ans[3], 'b')
plt.plot(ans[4], ans[5], 'g')
plt.plot(ans[6], ans[7], 'y')
plt.show()

dt = 0.000001
y = np.concatenate((y, velocity1(1, y)))

ans = []
for i in range(M):
	y = runge_kutta(y, 1, dt, velocity2)
	ans.append(y[0:2*n])

ans = np.transpose(np.array(ans))
plt.plot(ans[0], ans[1], 'c')
plt.plot(ans[2], ans[3], 'm')
plt.plot(ans[4], ans[5], 'k')
plt.plot(ans[6], ans[7], 'y')
plt.show()
