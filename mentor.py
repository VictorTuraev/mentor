import numpy as np
import time

r = np.array([[200., 200.],[200., 100.]])
v = np.array([[0.5, 0.], [-0.5, 0.]])
c = np.array([1., -1])
m = 1.
b = 1
dt = 0.05








class Vortices():
    def __init__(self, r, v, c, m, b, dt):
        self.r = np.copy(r)
        self.v = np.copy(v)
        self.c = np.copy(c)
        self.m = m
        self.b = b
        self.dt = dt
        self.n = np.shape(self.r)[0]
    def acceleration(self):
        n = self.n
        acc = np.zeros_like(r)
        for i in range(n):        
            for j in range(n):
                if (j != i) and (np.linalg.norm(self.r[i]-self.r[j]) != 0.):
                    acc[i] += self.c[i] * self.c[j] * (self.r[j]-self.r[i])  / (- self.m) / ( np.linalg.norm(self.r[i]-self.r[j])**2)
            acc[i,0] += self.v[i, 1]*self.b*self.c[i]/(-self.m)
            acc[i,1] -= self.v[i, 0]*self.b*self.c[i]/(-self.m)
            #Fx+=(c1 * c2 * (r1-r2)  /( np.linalg.norm(r1-r2)**2))*cos(a)#Fx=c[i]*c[j]*(x[i]-x[j])/((x[i]-x[j])**2+(y[i]-y[j])**2)
        return acc
    def move(self):
        acc = self.acceleration()
        self.r += self.v * dt + 0.5 * acc * dt**2
        self.v += acc * dt
        
        




vort = Vortices(r, v, c, m, b, dt)





import tkinter as tk

root = tk.Tk()
fr = tk.Frame(root)
root.geometry('800x600')
canv = tk.Canvas(root, bg='white')
canv.pack(fill=tk.BOTH, expand=1)

for i in range(vort.n):
    canv.create_oval(vort.r[i,0] - 10,
                vort.r[i,1] - 10,
                vort.r[i,0] + 10,
                vort.r[i,1] + 10,
                fill='green')




for i in range(100000):
    canv.delete(tk.ALL)
    vort.move()
    for i in range(vort.n):
        canv.create_oval(vort.r[i,0] - 10,
                vort.r[i,1] - 10,
                vort.r[i,0] + 10,
                vort.r[i,1] + 10,
                fill='green')
    if not(i % 10):
        print(vort.acceleration())
    canv.update()
    time.sleep(0.001)


root.mainloop()



