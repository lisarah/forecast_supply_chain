#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:17:30 2024

@author: huilih
"""
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import supply_chain_peak_gain as pg


backlog = 0.1 # backlog = alpha
A = np.array([[1, backlog], [0, 1-backlog]])
B = np.array([[0],[1]])

Bw = np.array([[-1], [0]])
   
noises = 2*np.random.rand(1000) - 1.
conv = 1
F_p = conv - backlog
F_i = conv**2/4/backlog
BF = np.array([[0,0], [-F_i, -F_p]])
A_f = A + BF
D = np.zeros((1,1))
C = np.array([[1, 0]])


alphas  = [0.5]
Q, Y, sigma, F_dp = pg.LMI_book(A_f, B, Bw, C, D, alphas[0])
peak_gains = [sigma]
iteration = 100
step_size = 1e-2
batch_num = 3
eps = 1e-5
for k in range(iteration):
    print(f'\r ---------     on step {k}/{iteration}     ----------- ', end = '')
    a_k = alphas[k]
    gk = pg.grad_est(a_k, batch_num, A_f, B, C, D, Bw)
    alphas.append(min(max(a_k  + step_size*gk, eps), 1))
    Q_k, Y_k, sigma_k, F_dp_k = pg.LMI_book(A_f, B, Bw, C, D, alphas[-1])
    peak_gains.append(sigma_k)
    if len(peak_gains) >=2 and abs(peak_gains[-2] - peak_gains[-1]) <=  1e-5:
        break
plt.figure()
plt.plot(peak_gains)
plt.yscale('log')
plt.show(block=False)


T = 100
min_sigma = []
seed = np.random.randint(1e8)
np.random.seed(seed)
A = np.random.rand(2,2)
B = np.random.rand(2,1)
C = np.random.rand(1,2)
# D = np.random.rand(1,1)



for alpha in range(T):
    print(f'\r {alpha} ', end = '')
    Q, Y, sigma, F_d = pg.LMI_book(A,B,Bw, C, D, (alpha+1)/T)
    min_sigma.append(sigma)
# Y.dot(np.linalg.inv(Q))
plt.figure()
plt.plot(min_sigma)
plt.grid()
plt.show(block=False)



# #----------- H2 gain solver ---------------#
# P, F2, F_d2 = LMI_H2(A_f, B, C, Bw)
# F_x = F2.dot(np.linalg.inv(P))# feedback =FP^{-1}x + F_d d
# #-----------------end H2 ------------------#
# hist_h2 = sim(A_f + B.dot(F_x), Bw, noises, F_d=F_d2)
# plot(hist_h2, C, 'H2 gain')
# #----------- peak gain solver ---------------#
# # solving it for one alpha
# Q, Y, sigma, F_dp = LMI_book(A_f, B, Bw, C, D, 0.8)
# F_x = Y.dot(np.linalg.inv(Q))# feedback = YQ^{-1}x + F_wd
# #--------------- end peak gain --------------#
# hist_peak = sim(A_f + B.dot(F_x), Bw, noises, F_d=F_dp)
# plot(hist_peak, C, 'peak gain')
# # hist = sim(A_f, D, noises, F_d=30)
# # plot(hist)

# Q, F, gamma = LMI_book(A, B, Bw, Dz)

# Q = cvx.Variable((2,2), symmetric=True)
# F_d = backlog*(backlog + F_p) + 10
# alpha = 0.001
# Bw = cvx.vstack((1, F_d))
# LMI = 1/alpha * Bw@Bw.T  - Q - 1/(1-alpha)*A_f@Q@A_f.T
# constraints = [LMI<<0, Q>>0]

# find_fd = cvx.Problem(cvx.Minimize(cvx.tr_inv(Q)), constraints)
# find_fd.solve(solver=cvx.MOSEK)
#------------ plot the resulting function curve -----------------------#
# T = 300
# min_sigma = []
# for alpha in range(T):
#     print(f'\r {alpha} ', end = '')
#     Q, Y, sigma, F_d = LMI_book(A,B,Bw,Dz, (alpha+1)/T)
#     min_sigma.append(sigma)
# # Y.dot(np.linalg.inv(Q))
# plt.figure()
# plt.plot(min_sigma)
# plt.grid()
# plt.show(block=False)
#--------------------end of plotting function --------------------------#

# D = Dz
# Q = cvx.Variable((2,2), symmetric=True)
# alpha = 0.001 #cvx.Variable(nonneg=True)
# Y = cvx.Variable((1,2))
# sigma = 10 # cvx.Variable(nonneg=True)
# LMI_1 = cvx.vstack((
#     cvx.hstack((-Q + alpha*Q, 0*np.ones((2,1)), Q@A.T+Y.T@B.T)), #,
#     cvx.hstack((0*np.ones((1,2)), -alpha*np.ones((1,1)), Bw.T)),
#     cvx.hstack((A@Q + B@Y, Bw, -Q))))
# LMI_2 = cvx.vstack((
#     cvx.hstack((sigma*Q, Q+ Y@D)),
#     cvx.hstack((Q + D@Y, -np.eye(2)))
#     ))






