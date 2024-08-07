from scipy.optimize import NonlinearConstraint, differential_evolution
import numpy as np

from BaseTask import BaseTask
from Opt3Dberry import Opt3Dberry
from ZH import ZH

hfov = 53.5

Z_0 = 100.0

delta_d = 2.0
delta_Z_0 = 3.0

w_screen = 960

L_10  = 10.0
wbox  = 15.0
hbox  = 15.0
s_1   = 1.5

#####################################

use_ZH = False
H_0 = 30.0

#####################################

alpha = np.deg2rad(hfov / 2.0)
L_20  = L_10

focal_length = w_screen / (2*np.tan(alpha))
T_min = Z_0 * Z_0 * delta_d / (focal_length * delta_Z_0)

print('focal_length = ', focal_length)
print('T_min = ', T_min)

##########################################

phi_1_0 = 150.0
phi_2_0 = 130.0
b_0     = hbox / 2.0

x0     = np.array([ np.deg2rad(phi_1_0), np.deg2rad(phi_2_0), b_0 ])
bounds = [(np.pi/2, np.pi), (np.pi/2, np.pi), (0, hbox) ]

####################################

bts = BaseTask(T_min, alpha, L_10, L_20)
o3db = Opt3Dberry(alpha, wbox, hbox, s_1)
zh = ZH(Z_0, H_0, alpha)

def cons_gen(i, j, k):
    def cons(x):
        cons_0 = bts.cons(x)
        add    = o3db.cons(x, i, j)
        add_1  = zh.cons(x, k)
        cons_0 = np.append(cons_0, add)
        cons_0 = np.append(cons_0, add_1)
        return cons_0
    return cons

def lb_gen(i, j, k):
    lb_0  = bts.lb()
    add   = o3db.lb(i, j)
    add_1 = zh.lb(k)
    lb_0 = np.append(lb_0, add)
    lb_0 = np.append(lb_0, add_1)
    return lb_0

def ub_gen(i, j, k):
    ub_0 = bts.ub()
    add = o3db.ub(i, j)
    add_1 = zh.ub(k)
    ub_0 = np.append(ub_0, add)
    ub_0 = np.append(ub_0, add_1)
    return ub_0

####################################

global_opt_vals = np.array([0,0,0, 1e9])

if (use_ZH):
    k_min = 0
    k_max = 5
    k_step = 1
else:
    k_min = -1
    k_max = -2
    k_step = -1

for i in range(0,2):
    for j in range(0,2):
        for k in range(k_min, k_max, k_step):
            print('-------------', i, j, k, '---------------')
            nlc = NonlinearConstraint(cons_gen(i,j,k), lb_gen(i,j,k), ub_gen(i,j,k))
            res = differential_evolution(o3db.fun, bounds, constraints=nlc, x0=x0)
            if (res.success):
                print( np.rad2deg(res.x[0]), np.rad2deg(res.x[1]), res.x[2] )
                print('perimeter =', res.fun)
                print('base = ', bts.base(res.x))
                print('delta_phi = ', np.rad2deg(res.x[0]) - np.rad2deg(res.x[1]))
                print('box = ', o3db.get_box(res.x))
                if (res.fun < global_opt_vals[3]):
                    global_opt_vals[0] = res.x[0]
                    global_opt_vals[1] = res.x[1]
                    global_opt_vals[2] = res.x[2]
                    global_opt_vals[3] = res.fun

print('======================================================')
print('Optimized values (global maximum):')
print('phi_1 =', np.rad2deg(global_opt_vals[0]), ', phi_2 =', np.rad2deg(global_opt_vals[1]), \
      ', b =', global_opt_vals[2])
print('perimeter =', global_opt_vals[3])
print('base = ', bts.base(global_opt_vals))
print('delta_phi = ', np.rad2deg(global_opt_vals[0]) - np.rad2deg(global_opt_vals[1]))
print('box = ', o3db.get_box(global_opt_vals))