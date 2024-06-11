import numpy as np
import sys

alpha = 35.0
T_min = 6.0
L_1   = 8.0
u     = 1.5
L_2   = L_1

step = 0.01

####################################3

alpha = np.deg2rad(alpha)
sin_a = np.sin(alpha)
pi    = np.pi

################

def G(b):
	L_1_   = L_1/b
	L_2_   = L_2/b
	u_     = u/b
	T_min_ = T_min/b
	
	acs_1   = np.arccos(sin_a/(L_1_))
	acs_2   = np.arccos(sin_a/(L_2_))
	actg_u_ = pi/2.0 - np.arctan(u_)
	acs_u_  = np.arccos(sin_a/np.sqrt(1+np.power(u_, 2.0)))
	
	L_1_min = sin_a / np.cos(pi/4.0 + alpha - 0.5*actg_u_)
	if ( L_1_ < L_1_min ):
	    return []
	
	phi1_min = pi + alpha - acs_1
	phi1_max = 0.75*pi + 0.5*actg_u_
	phi2_min = pi - alpha - acs_2
	phi2_max = pi/2.0 + 0.5*acs_u_ + 0.5*(actg_u_ - alpha)
	
	a = np.arcsin(T_min_/2.0)
	
	if ((phi1_max < (2.0*a + pi/2.0)) or (phi2_max < (a + pi/2.0)) or \
		(phi1_min > (2.0*phi2_max - pi/2.0)) or (phi1_max < (phi2_min + a)) ):
	    res = []
	elif (phi1_max < (2.0*phi2_min - pi/2.0)):
	    res = [phi1_max, phi2_min, 'solid']
	elif (phi1_max > (2.0*phi2_max - pi/2.0)):
	    res = [2.0*phi2_max - pi/2.0, phi2_max, 'dashed']
	else:
	    res = [phi1_max, 0.5*phi1_max + pi/4.0, 'dotted']
	
	return res
#############################

p1_min = pi
p1_max = pi/2
p2_min = pi
p2_max = pi/2.0

def F(b):
    p = G(b)
    val = 0.0
    if ( len(p) > 0 ):
        val = 2.0*b*np.sin(p[0] - p[1])
        global p1_min, p1_max, p2_min, p2_max
        if ( p[0] < p1_min ):
            p1_min = p[0]
        if ( p[0] > p1_max ):
            p1_max = p[0]
        if ( p[1] < p2_min ):
            p2_min = p[1]
        if ( p[1] > p2_max ):
            p2_max = p[1]
    return (p, val)

#################################

b_min_1 = u * np.tan(np.max([0, 2*alpha - pi/2.0]))
b_min_2 = T_min / 2.0
b_min = np.max([b_min_1, b_min_2]) + 0.0001
b_max = np.min([L_1, L_2]) / sin_a - 0.0001

if (b_min > b_max):
    print("EMPTY b")
    sys.exit(1)


N = int(np.round((b_max - b_min)/step))

xs = []
ps = []
ys = []

for i in range(0,N):
    b = b_min + i*step
    xs.append(b)
xs.append(b_max)

max_val = 0.0
res = [0,0,0]

for b in xs:
    (p, Fb) = F(b)
    ps.append(p)
    ys.append(Fb)
    if ( Fb > max_val ):
        max_val = Fb
        res[0] = b
        res[1] = np.rad2deg(p[0])
        res[2] = np.rad2deg(p[1])

print('phi_1 =', res[1], ', phi_2 =', res[2], ', b =', res[0])
print('base =', max_val)