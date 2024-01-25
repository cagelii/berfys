def step(h,f,t,y):
	
	k = h*f(t,y)
	return y+h*f(t+h/2,y+k/2)