from scipy.optimize import minimize, NonlinearConstraint, differential_evolution, shgo, \
                           LinearConstraint, milp
import numpy as np

alpha = 35.0
T_min = 6.0
L_1   = 8.0
u     = 1.5
L_2   = L_1

step = 0.01

alpha = np.deg2rad(alpha)
sina = np.sin(alpha)

def F(b):

    c = np.array([ -1, 1 ])
    A = np.array([ [1, 0], [0, 1], [1, -1], [1, -1], [1, 0], [0, 1], [-1, 2], [1, 0], [0, 1] ])

    temp_1  = np.arcsin(T_min / (2*b))
    temp_21 = np.arccos(sina / (L_1 / b))
    temp_22 = np.arccos(sina / (L_2 / b))
    temp_3  = 0.5*np.arctan(b/u)
    temp_4  = 0.5*np.arccos(sina / np.sqrt(1+np.power(u/b, 2.0)) )

    lb  = np.array([ np.pi/2 + alpha, \
                     np.pi/2, \
                     0, \
                     temp_1, \
                     np.pi + alpha - temp_21, \
                     np.pi - alpha - temp_22, \
                     np.pi/2, \
                     np.pi/2, \
                     np.pi/2 \
                 ])

    ub  = np.array([ np.pi, \
                     np.pi, \
                     np.pi/2, \
                     np.pi - temp_1, \
                     np.pi + alpha + temp_21, \
                     np.pi - alpha + temp_22, \
                     3*np.pi/2, \
                     3*np.pi/4 + temp_3, \
                     np.pi/2 + temp_4 + temp_3 - 0.5*alpha \
                 ])

    lc = LinearConstraint(A, lb, ub)
    res = milp(c=c, constraints=lc)
    return res.success, res.x

##########################################

b_min = T_min/2.0
b_max = np.minimum(L_1,L_2)/sina

N = int(np.round((b_max - b_min)/step))

bs = []
for i in range(0,N):
    bs.append(b_min + i*step)
bs.append(b_max)

max_val = 0.0
res = [0,0,0]

for b in bs:
    succ, phi_12 = F(b)
    if ( succ ):
        base = 2*b*np.sin(phi_12[0] - phi_12[1])
        if ( base > max_val ):
            max_val = base
            res[0] = phi_12[0]
            res[1] = phi_12[1]
            res[2] = b

print('phi_1 =', np.rad2deg(res[0]), ', phi_2 =', np.rad2deg(res[1]), ', b =', res[2])
print('base =', max_val)