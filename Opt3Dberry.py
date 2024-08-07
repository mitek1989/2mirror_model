import numpy as np

huge_val = 1e9

class Opt3Dberry:

    def __init__(self, alpha, wbox, hbox, s_1):
        self.alpha = alpha
        self.sina = np.sin(alpha)
        self.ctga = 1.0 / np.tan(alpha)

        self.wbox = wbox
        self.hbox = hbox
        self.s_1 = s_1

    def AC(self, x):
        phi_1 = x[0]
        phi_2 = x[1]
        b     = x[2]
        
        X_A = b / (-self.ctga - np.tan(phi_1))
        Z_A = -X_A * self.ctga

        X_C = b / ( self.ctga - np.tan(phi_2))
        Z_C =  X_C * self.ctga
        
        return [X_A, Z_A], [X_C, Z_C]
        
    def get_box(self, x):
        A, C = self.AC(x)
        return C[0] - A[0], A[1]

    def line_coeffs(self, P1, P2):
        # p*x + q*y + r = 0
        p = P2[1] - P1[1]
        q = P1[0] - P2[0]
        r = P1[1]*P2[0] - P1[0]*P2[1]
        return p, q, r
    
    def cons(self, x, i, j):
        phi_1 = x[0]
        phi_2 = x[1]
        b     = x[2]
        
        A, C = self.AC(x)
        B = [0, b]
        T1 = [ -b * np.sin(2*phi_2), b * (1 + np.cos(2*phi_2)) ]
        T2 = [ -b * np.sin(2*phi_1), b * (1 + np.cos(2*phi_1)) ]
        
        # l41: через точки T2 и B, p41*x + q41*z + r41 = 0
        p41, q41, r41 = self.line_coeffs(T2, B)
        
        # l32: через точки T1 и C, p32*x + q32*z + r32 = 0
        p32, q32, r32 = self.line_coeffs(T1, C)
        
        arr = np.array([ \
            C[0] - A[0], \
            A[1], \
        ])
        
        con_1 = np.array([phi_1]) if (i == 0) else np.array([phi_1, -r41/p41])
        con_2 = np.array([phi_2]) if (j == 0) else np.array([phi_2, -r32/p32])
        
        arr = np.append(arr, con_1)
        arr = np.append(arr, con_2)
        return arr

    def lb(self, i, j):
        arr = np.array([0, 0])
        
        add_1 = np.array([np.pi/2]) if (i == 0) else np.array([3*np.pi/4, -huge_val])
        add_2 = np.array([np.pi/2]) if (j == 0) else np.array([3*np.pi/4 - self.alpha/2, -huge_val])
        
        arr = np.append(arr, add_1)
        return np.append(arr, add_2)

    def ub(self, i, j):
        arr = np.array([self.wbox, self.hbox])
        
        add_1 = np.array([3*np.pi/4])                if (i == 0) else np.array([np.pi, -self.s_1])
        add_2 = np.array([3*np.pi/4 - self.alpha/2]) if (j == 0) else np.array([np.pi, -self.s_1])
        
        arr = np.append(arr, add_1)
        return np.append(arr, add_2)
    
    def fun(self, x): # периметр
        phi_1 = x[0]
        phi_2 = x[1]
        b     = x[2]
        
        A, C = self.AC(x)
        return 2*( (C[0] - A[0]) + A[1] )
