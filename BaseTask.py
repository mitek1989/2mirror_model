import numpy as np

huge_val = 1e9

class BaseTask:

    def __init__(self, T_min, alpha, L_10, L_20):
        self.T_min = T_min
        self.alpha = alpha
        self.L_10 = L_10
        self.L_20 = L_20
        self.sina = np.sin(alpha)
        
    def base(self, x):
        phi_1 = x[0]
        phi_2 = x[1]
        b     = x[2]
        return 2*b*np.sin(phi_1 - phi_2)
        
    def L_1(self, x):
        phi_1 = x[0]
        b     = x[2]
        return -b*self.sina/np.cos(phi_1 - self.alpha)

    def L_2(self, x):
        phi_2 = x[1]
        b     = x[2]
        return -b*self.sina/np.cos(phi_2 + self.alpha)

    def repair(self, val, minv, maxv):
        if (val < minv):
            val = minv
        elif (val > maxv):
            val = maxv
        return val
    
    def cons(self, x):
        phi_1 = x[0]
        phi_2 = x[1]
        b     = x[2]
    
        return np.array([
            phi_1, \
            phi_1 - phi_2, \
            phi_2, \
            b, \
            self.base(x), \
            self.L_1(x), \
            self.L_2(x), \
            2*phi_2 - phi_1, \
        ])

    def lb(self):
        return np.array([
            np.pi/2 + self.alpha, \
            0, \
            np.pi/2, \
            0, \
            self.T_min, \
            0, \
            0, \
            np.pi/2, \
        ])

    def ub(self):
        return np.array([
            np.pi, \
            np.pi/2, \
            np.pi, \
            huge_val, \
            huge_val, \
            self.L_10, \
            self.L_20, \
            3*np.pi/2, \
        ])
