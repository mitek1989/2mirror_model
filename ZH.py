import numpy as np

huge_val = 1e5

class ZH:

    def __init__(self, Z_0, H_0, alpha):
        self.Z_0 = Z_0
        self.H_0 = H_0
        self.alpha = alpha
        self.sina = np.sin(alpha)
        
    def f1( self, delta_phi, b ):
        return 2*self.Z_0*np.tan(delta_phi) - 2*b*np.sin(delta_phi)

    def f2( self, delta_phi, b ):
        return b*np.cos(delta_phi)*(1 + self.sina/np.sin(2*delta_phi - self.alpha))

    def f3( self, delta_phi, b ):
        return 2*self.Z_0*np.tan(self.alpha - delta_phi) + 2*b*np.sin(delta_phi)

    def f4( self, delta_phi, b ):
        return b*np.sin(delta_phi) / np.tan(delta_phi - self.alpha)

    def cons(self, x, k):
        if   (k == -1):
            return np.array([])
        
        phi_1 = x[0]
        phi_2 = x[1]
        b     = x[2]
        
        d = phi_1 - phi_2
        
        if   (k == 0):
            return np.array([ d, self.f1(d, b) ])
        elif (k == 1):
            return np.array([ d, self.f2(d, b), self.f1(d, b) ])
        elif (k == 2):
            return np.array([ d, self.f2(d, b), self.f3(d, b) ])
        elif (k == 3):
            return np.array([ d, self.f2(d, b), self.f1(d, b) ])
        elif (k == 4):
            return np.array([ d, self.f2(d, b), self.f3(d, b), self.f4(d, b) ])

    def lb(self, k):
        if   (k == -1):
            return np.array([])
        elif (k == 0):
            return np.array([ 0,            self.H_0 ])
        elif (k == 1):
            return np.array([ self.alpha/2, self.Z_0, self.H_0 ])
        elif (k == 2):
            return np.array([ self.alpha/2, 0,        self.H_0 ])
        elif (k == 3):
            return np.array([ self.alpha,   self.Z_0, self.H_0 ])
        elif (k == 4):
            return np.array([ self.alpha,   0,        self.H_0, self.Z_0 ])

    def ub(self, k):
        if   (k == -1):
            return np.array([])
        if   (k == 0):
            return np.array([ self.alpha/2, huge_val ])
        elif (k == 1):
            return np.array([ self.alpha,   huge_val, huge_val ])
        elif (k == 2):
            return np.array([ self.alpha,   self.Z_0, huge_val ])
        elif (k == 3):
            return np.array([ np.pi/2,      huge_val, huge_val ])
        elif (k == 4):
            return np.array([ np.pi/2,      self.Z_0, huge_val, huge_val ])
