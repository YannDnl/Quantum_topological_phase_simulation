import simulation
import time as t
import numpy as np

from parameters import *

q = Q
p = P
n = N
h = [H for _ in range(n)]
e = [E for _ in range(n)]
d = 1
r = np.array([[RXX, RXY, RXZ],
              [RYX, RYY, RYZ],
              [RZX, RZY, RZZ]])
m2 = M1
m = [m2, .3]
m_sur_d_min = 0
m_sur_d_max = 2


n_points = 11
k_points = 18
l_points = 18
l = 10

############################################
# time comparison parallel vs serial
############################################

#if __name__ == '__main__':
#    ss = t.time()
#    c = simulation.computePsi(n, m, h, e, r, q, p)
#    es = t.time()
#    sp = t.time()
#    c = simulation.computePsiParallel(n, m, h, e, r, q, p)
#    ep = t.time()
#    print(f"Serial: {es - ss}, Parallel: {ep - sp}")

#if __name__ == '__main__':
#    ss = t.time()
#    c = simulation.computeC(n, m, h, e, r, q, p)
#    es = t.time()
#    print(f"C in: {es - ss}")


#if __name__ == '__main__':
#    ss = t.time()
#    c = simulation.plotCvsM(m_sur_d_min, m_sur_d_max, q, p, n_points)
#    es = t.time()
#    print(f"Serial: {es - ss} s")
#    for k in range(5, 13):
#        sp = t.time()
#        c = simulation.plotCvcMParallel(m_sur_d_min, m_sur_d_max, q, p, n_points, k)
#        ep = t.time()
#        print(f"Parallel with {k} processes: {ep - sp} s")

#if __name__ == '__main__':
#    ss = t.time()
#    c = simulation.plotPhase(k_points, l_points, q, p)
#    es = t.time()
#    sp = t.time()
#    c = simulation.plotPhaseParallel(k_points, l_points, q, p)
#    ep = t.time()
#    print(f"Serial: {es - ss}, Parallel: {ep - sp}")

############################################
# debug stuff
############################################

#simulation.plotSteps(n, m, h, e, r, q, p)

############################################
# compute and plot C
############################################

if __name__=='__main__':
#    c = simulation.computeC(n, m, h, e, r, q, p)
#    print(f"Chern number: {c}")

#    simulation.plotCvsMParallel(m_sur_d_min, m_sur_d_max, q, p, n_points)

#    simulation.plotSingleLineParallel(l, m2, q, p)

    simulation.plotPhaseParallel(k_points, l_points, q, p)