import graphics as gr
import math
win = gr.GraphWin("mayatnik", 500, 500)
dt = 10
a = 0
v = 500
r = 20
l=200
x=250
y=10+l*math.cos(a)

nit = gr.Line(gr.Point(250, 10), gr.Point(x, y))
nit.draw(win)
circle = gr.Circle(gr.Point(x, y), r)
circle.setFill("black")
circle.draw(win)

def may():
	global x, y, circle
	x, y = mv(a)
	circle = gr.Circle(gr.Point(x, y), r)
	circle.setFill("black")
	circle.draw(win)

def change(a, v):
	k=v*dt/r
	b=-10*math.sin(a)
	a=a+k
	v=v+b
	return a
	
def nt():
	global x, y, nit
	x, y=mv(a)
	nit = gr.Line(gr.Point(250, 10), gr.Point(x, y))
	nit.draw(win)
	gr.time.sleep(0.2)
	
def mv(a):
	global nit, circle
	a = change(a, v)
	x=l*math.sin(a)+250
	y=l*math.cos(a)+10
	nit.undraw()
	gr.time.sleep(0.2)
	circle.move(x, y)
	return x, y
def main():
	for i in range(500):
		may()
		nt()
		change(a, v)
		mv(a)
		gr.time.sleep(0.2)
main()


