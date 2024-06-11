from scipy.optimize import minimize, NonlinearConstraint, differential_evolution, shgo
import numpy as np

alpha = 53.5 / 2.0

phi_1_0 = 180.0 - 35.0
phi_2_0 = 180.0 - 50.0
b_0     = 3.1

##################################

alpha = np.deg2rad(alpha)
sina = np.sin(alpha)
ctga = 1.0 / np.tan(alpha)

def AC(x):
    phi_1 = x[0]
    phi_2 = x[1]
    b     = x[2]
    
    X_A = b / (-ctga - np.tan(phi_1))
    X_C = b / ( ctga - np.tan(phi_2))
    
    Z_A = -X_A * ctga
    Z_C =  X_C * ctga
    return np.array([X_A, Z_A]), np.array([X_C, Z_C])

def fun(x):
    phi_1 = x[0]
    phi_2 = x[1]
    b     = x[2]
    return -2.0*b*np.sin(phi_1 - phi_2)

##########################################

x0 = np.array([ np.deg2rad(phi_1_0), np.deg2rad(phi_2_0), b_0 ])

print('Initial values:')
print('phi_1 =', phi_1_0, ', phi_2 =', phi_2_0, ', b =', b_0)
print('base =', -fun(x0))

A, C = AC(x0)
Z_max = A[1]
u1 = C[0] - A[0]
print('--------------------------')
print('Z_max = ', Z_max, ', u1 = ', u1)
print('--------------------------')

###########################################

def cons(x):
    phi_1 = x[0]
    phi_2 = x[1]
    b     = x[2]
    
    temp_3  = 0.5*np.arctan(b/u1)
    temp_4  = 0.5*np.arccos(sina / np.sqrt(1+np.power(u1/b, 2.0)) )
    
    A, C = AC(x)
    
    return [   phi_1, \
               phi_1 - phi_2, \
               phi_2, \
               2*phi_2 - phi_1, \
               phi_1 - temp_3, \
               phi_2 - temp_4 - temp_3 + 0.5*alpha, \
               A[1], \
               C[0] - A[0] \
             ]

lb  = np.array([   np.pi/2 + alpha, \
                   0, \
                   np.pi/2, \
                   np.pi/2, \
                   np.pi/4, \
                   0, \
                   0, \
                   0 \
                 ])

ub  = np.array([   np.pi, \
                   np.pi/2, \
                   np.pi, \
                   3*np.pi/2, \
                   3*np.pi/4, \
                   np.pi/2, \
                   Z_max, \
                   u1 \
                 ])

nlc = NonlinearConstraint(cons, lb, ub)

b_min = 0
b_max = Z_max

bounds = [(np.pi/2, np.pi), (np.pi/2, np.pi), (b_min, b_max) ]

res_1 = minimize(fun, x0, constraints=nlc)

print('Optimized values (local maximum):')
print('phi_1 =', np.rad2deg(res_1.x[0]), ', phi_2 =', np.rad2deg(res_1.x[1]), ', b =', res_1.x[2])
print('base =', -res_1.fun)

res_2 = differential_evolution(fun, bounds, constraints=nlc, x0=x0)

print('Optimized values (global maximum):')
print('phi_1 =', np.rad2deg(res_2.x[0]), ', phi_2 =', np.rad2deg(res_2.x[1]), ', b =', res_2.x[2])
print('base =', -res_2.fun)