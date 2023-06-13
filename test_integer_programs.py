import numpy as np

x = np.random.uniform(low=-4,high=4,size=5)
k=1

a = 1/(1+np.exp(-x))
# y = np.random.uniform(low=-4,high=4,size=5)
# b = 1/(1+np.exp(-3*y))
b = 1-a

a_const = a + (1-b-a)*(1/2+1/2*np.tanh(k*(a-1+b)))+(1-b-a)*(1/2+1/2*np.tanh(k*(1-b-a)))
b_const = b + (1-a-b)*(1/2+1/2*np.tanh(k*(b-1+a)))+(1-a-b)*(1/2+1/2*np.tanh(k*(1-a-b)))
print(a,a_const)
print(b,b_const)

print(a_const+b_const)
