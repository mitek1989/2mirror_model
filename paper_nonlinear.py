from scipy.optimize import minimize, NonlinearConstraint, differential_evolution, shgo
import numpy as np

alpha = 35.0
T_min = 6.0
L_1   = 8.0
u     = 1.5
L_2   = L_1

alpha = np.deg2rad(alpha)
sina = np.sin(alpha)

def fun(x):
    phi_1 = x[0]
    phi_2 = x[1]
    b     = x[2]
    return -2.0*b*np.sin(phi_1 - phi_2)

def cons(x):
    phi_1 = x[0]
    phi_2 = x[1]
    b     = x[2]
    
    temp_1  = np.arcsin(T_min / (2*b))
    temp_21 = np.arccos(sina / (L_1 / b))
    temp_22 = np.arccos(sina / (L_2 / b))
    temp_3  = 0.5*np.arctan(b/u)
    temp_4  = 0.5*np.arccos(sina / np.sqrt(1+np.power(u/b, 2.0)) )
    
    return [   phi_1, \
               phi_1 - phi_2, \
               phi_2, \
               phi_1 - phi_2 - temp_1, \
               phi_1 - phi_2 + temp_1, \
               phi_1 + temp_21, \
               phi_1 - temp_21, \
               phi_2 + temp_22, \
               phi_2 - temp_22, \
               2*phi_2 - phi_1, \
               phi_1 - temp_3, \
               phi_2 - temp_4 - temp_3 + 0.5*alpha, \
             ]

lb  = np.array([   np.pi/2 + alpha, \
                   0, \
                   np.pi/2, \
                   0, \
                   0, \
                   np.pi + alpha, \
                   0, \
                   np.pi - alpha, \
                   0, \
                   np.pi/2, \
                   np.pi/4, \
                   0 \
                 ])

ub  = np.array([   np.pi, \
                   np.pi/2, \
                   np.pi, \
                   np.pi/2, \
                   np.pi, \
                   3*np.pi/2, \
                   np.pi + alpha, \
                   3*np.pi/2, \
                   np.pi - alpha, \
                   3*np.pi/2, \
                   3*np.pi/4, \
                   np.pi/2 \
                 ])

##########################################

nlc = NonlinearConstraint(cons, lb, ub)

b_min = T_min/2.0
b_max = np.minimum(L_1,L_2)/sina

phi_1_0 = np.deg2rad(180.0)
phi_2_0 = np.deg2rad(135.0)
b_0     = (b_min + b_max) / 2.0

x0 = np.array([ phi_1_0, phi_2_0, b_0 ])

bounds = [(np.pi/2, np.pi), (np.pi/2, np.pi), (b_min, b_max) ]

res_1 = minimize(fun, x0, constraints=nlc)
print('Local maximum:')
print('phi_1 =', np.rad2deg(res_1.x[0]), ', phi_2 =', np.rad2deg(res_1.x[1]), ', b =', res_1.x[2])
print('base =', -res_1.fun)

res_2 = differential_evolution(fun, bounds, constraints=nlc, x0=x0)
print('Global maximum:')
print('phi_1 =', np.rad2deg(res_2.x[0]), ', phi_2 =', np.rad2deg(res_2.x[1]), ', b =', res_2.x[2])
print('base =', -res_2.fun)
