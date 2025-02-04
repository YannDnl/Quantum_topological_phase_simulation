import simulation

n = 1
m = 0
d = 1
q = 100
p = 200
m_sur_d_min = 0
m_sur_d_max = 2
n_points = 5
k_points = 10
l_points = 12

#simulation.plotPsiDot(m, d, q, p)
#simulation.plotCvsM(m_sur_d_min, m_sur_d_max, q, p, n_points)
simulation.plotPhase(k_points, l_points, q, p)