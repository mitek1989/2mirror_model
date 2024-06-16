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

def T1T2(x):
    phi_1 = x[0]
    phi_2 = x[1]
    b     = x[2]
    
    X_T1 = -b * np.sin(2*phi_2)
    Z_T1 =  b * (1 + np.cos(2*phi_2))

    X_T2 = -b * np.sin(2*phi_1)
    Z_T2 =  b * (1 + np.cos(2*phi_1))

    return np.array([X_T1, Z_T1]), np.array([X_T2, Z_T2])

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

def l41_l32(x, C):

    T1, T2 = T1T2(x)
    b = x[2]

    # l41: через точки T2 и B, p41*x + q41*z + r41 = 0
    # l32: через точки T1 и C, p32*x + q32*z + r32 = 0
    
    p41 = -(b - T2[1])
    # q41 =   - T2[0]
    r41 =  T2[0]*b

    p32 = -(C[1] - T1[1])
    # q32 =   C[0] - T1[0]
    r32 =  T1[0]*C[1] - T1[1]*C[0]
    return p41, r41, p32, r32

def get_cons_0(x):
    phi_1 = x[0]
    phi_2 = x[1]
    b     = x[2]
    
    A, C   = AC(x)

    return A, C, \
        [   phi_1, \
            phi_1 - phi_2, \
            phi_2, \
            2*phi_2 - phi_1, \
            A[1], \
            C[0] - A[0] \
        ]

def get_lb_0():
    return np.array([ \
        np.pi/2 + alpha, \
        0, \
        np.pi/2, \
        np.pi/2, \
        0, \
        0 \
    ])

def get_ub_0():
    return np.array([ \
        np.pi, \
        np.pi/2, \
        np.pi, \
        3*np.pi/2, \
        Z_max, \
        u1 \
    ])

####################################

def cons_4(x):
    phi_1 = x[0]
    phi_2 = x[1]
    A, C, cons_0 = get_cons_0(x)
    p41, r41, p32, r32 = l41_l32(x, C)
    add = np.array([p41 * A[0] + r41, phi_1, p32 * A[0] + r32, phi_2])
    return np.append(cons_0, add)

def cons_gen(i, j):
    def cons(x):
        phi_1 = x[0]
        phi_2 = x[1]
        A, C, cons_0 = get_cons_0(x)
        p41, r41, p32, r32 = l41_l32(x, C)
        con_1 = np.array([phi_1]) if (i == 0) else np.array([p41 * A[0] + r41, phi_1])
        con_2 = np.array([phi_2]) if (j == 0) else np.array([p32 * A[0] + r32, phi_2])
        cons_0 = np.append(cons_0, con_1)
        return np.append(cons_0, con_2)
    return cons

def lb(i, j):
    lb_0 = get_lb_0()
    add_1 = np.array([np.pi/2]) if (i == 0) else np.array([0, 3*np.pi/4])
    add_2 = np.array([np.pi/2]) if (j == 0) else np.array([0, 3*np.pi/4 - alpha/2])
    lb_0 = np.append(lb_0, add_1)
    return np.append(lb_0, add_2)

def ub(i, j):
    ub_0 = get_ub_0()
    add_1 = np.array([3*np.pi/4])           if (i == 0) else np.array([1e9, np.pi])
    add_2 = np.array([3*np.pi/4 - alpha/2]) if (j == 0) else np.array([1e9, np.pi])
    ub_0 = np.append(ub_0, add_1)
    return np.append(ub_0, add_2)

b_min = 0
b_max = Z_max

bounds = [(np.pi/2, np.pi), (np.pi/2, np.pi), (b_min, b_max) ]

local_opt_vals = np.zeros(4)
global_opt_vals = np.zeros(4)

for i in range(0,2):
    for j in range(0,2):
        nlc = NonlinearConstraint(cons_gen(i,j), lb(i,j), ub(i,j))
        res_1 = minimize(fun, x0, constraints=nlc)
        res_2 = differential_evolution(fun, bounds, constraints=nlc, x0=x0)
        
        if (res_1.fun < local_opt_vals[3]):
            local_opt_vals[0] = res_1.x[0]
            local_opt_vals[1] = res_1.x[1]
            local_opt_vals[2] = res_1.x[2]
            local_opt_vals[3] = res_1.fun

        if (res_2.fun < global_opt_vals[3]):
            global_opt_vals[0] = res_2.x[0]
            global_opt_vals[1] = res_2.x[1]
            global_opt_vals[2] = res_2.x[2]
            global_opt_vals[3] = res_2.fun

print('Optimized values (local maximum):')
print('phi_1 =', np.rad2deg(local_opt_vals[0]), ', phi_2 =', np.rad2deg(local_opt_vals[1]), ', b =', local_opt_vals[2])
print('base =', -local_opt_vals[3])
        
print('Optimized values (global maximum):')
print('phi_1 =', np.rad2deg(global_opt_vals[0]), ', phi_2 =', np.rad2deg(global_opt_vals[1]), ', b =', global_opt_vals[2])
print('base =', -global_opt_vals[3])