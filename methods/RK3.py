def step(h,f,t,y):
	
	k1 = f(t,y)
	k2 = f(t+h/2,y+k1/2)
	k3 = f(t+h,y-k1+2*k2)
	return y+h/6*(k1+4*k2+k3)
