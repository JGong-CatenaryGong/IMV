import scipy.integrate as spi
import numpy as np

def readdat(filename):
    data = np.loadtxt(filename)
    return data

pt = readdat('pt.dat')
print(pt / pt[0])

pt_V = spi.simpson(pt / pt[0], dx=0.1)
print(pt_V / 4.9)

ct = readdat('ct.dat')
print(spi.simpson(ct / ct[0], dx=0.1) / 4.9)