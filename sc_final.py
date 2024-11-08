#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:19:09 2024

@author: huilih
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx




def sim(A, B, Bw, noises, F_d = None, init_state=None):
    
    if init_state is None:
        init_state = 10*np.zeros((3,1))
        # init_state = 10*np.random.rand(3,1)
        # print(f' initial state is {init_state}')
    if F_d is not None:
        Bw = Bw + F_d
    
    hist = [init_state]
    for d_k in noises:
        x_k = hist[-1]
        hist.append(np.squeeze(A.dot(x_k)) + Bw.dot(d_k))
        # print('   another noise    ')
        # print(f'Bw d = {Bw.dot(d_k).T}')
        # print(f'(A+BF)x = {np.squeeze(A.dot(x_k))}')
        # print(A.dot(x_k) + Bw.dot(d_k))
        # print('--------end ------------')
    return hist

def plot(hist, F, noises, F_d, title=None):
    cols = ['order', 'inventory']
    plt.figure()
    obs_hist = [F.dot(x_k)[0]+ F_d*n_k[1] for x_k, n_k in zip(hist, noises) ]# 
    obs_hist2 = [F.dot(x_k)[0] for x_k in hist]#
    # obs_hists = [obs_hist, obs_hist2]
    print(obs_hist)
    if title is not None:
        plt.title(title)
    plt.plot(obs_hist, label=cols[0],alpha=0.5)
    plt.plot(obs_hist2, label=cols[1],alpha=0.5)
    plt.grid(); plt.legend();
    plt.show(block=False)
    



# def LMI_book(A, B, Bw, alpha, eps_f):
#     c_dim = 2
#     Q = cvx.Variable((2,2), symmetric=True)
#     Y = cvx.Variable((1,2))
#     F_d = cvx.Variable()
#     F_w = cvx.hstack((0,F_d))
#     BFw = cvx.vstack((cvx.hstack((-1,0)), F_w))
#     F_w = cvx.vstack(cvx.hstack((0,F_d)))
#     # cvx_Bw = Bw + BFw# Bw + BFw
#     gamma = cvx.Variable(nonneg=True)
#     sigma = cvx.Variable(nonneg=True) # works at 1750 with F_d, 2500 without F_d
#     LMI_1 = cvx.vstack((
#         cvx.hstack((-Q + alpha*Q, 0*np.ones((2,2)), Q@A.T+Y.T@B.T)), #,
#         cvx.hstack((0*np.ones((2,2)), -alpha/2*np.array([[1,0], [0,1/(1+eps_f)**2]]), BFw.T)),
#         cvx.hstack((A@Q + B@Y, BFw, -Q))))

#     LMI_2 = cvx.vstack((
#         cvx.hstack((-Q, 0*np.ones((2,2)), Y.T)), # 
#         cvx.hstack((0*np.ones((2,2)), -(gamma - sigma)/2*np.array([[1,0], [0,1/(1+eps_f)**2]]), F_w)),
#         cvx.hstack((Y, F_w.T, -cvx.vstack((sigma,))))        # 
#         ))
#     constraints = [Q>>0, LMI_1<<0, LMI_2<<0] # 
#     find_f = cvx.Problem(cvx.Minimize(gamma), constraints=constraints)
#     find_f.solve(solver=cvx.MOSEK) #,verbose=True
    
#     return Q.value, Y.value, sigma.value, BFw.value

# def LMI_no_forecast(A, B, Bw, alpha, perfect_forecast=True):
#     Q = cvx.Variable((2,2), symmetric=True)
#     Y = cvx.Variable((1,2))
#     F_d = cvx.Variable()
#     var = F_d if perfect_forecast else 0
#     B_w = cvx.vstack((-1, var))
#     # cvx_Bw = Bw + BFw# Bw + BFw
#     gamma = cvx.Variable(nonneg=True)
#     sigma = cvx.Variable(nonneg=True) # works at 1750 with F_d, 2500 without F_d
#     LMI_1 = cvx.vstack((
#         cvx.hstack((-Q + alpha*Q, 0*np.ones((2,1)), Q@A.T+Y.T@B.T)), #,
#         cvx.hstack((0*np.ones((1,2)), -alpha*np.eye(1), B_w.T)),
#         cvx.hstack((A@Q + B@Y, B_w, -Q))))

#     LMI_2 = cvx.vstack((
#         cvx.hstack((-Q, 0*np.ones((2,1)), Y.T)), # 
#         cvx.hstack((0*np.ones((1,2)), -(gamma - sigma)*np.eye(1), F_d*np.ones((1,1)))),
#         cvx.hstack((Y, F_d*np.ones((1,1)), -sigma*np.eye(1)))        # 
#         ))
#     constraints = [Q>>0, LMI_1<<0, LMI_2<<0] # 
#     find_f = cvx.Problem(cvx.Minimize(gamma), constraints=constraints)
#     find_f.solve(solver=cvx.MOSEK) #,verbose=True
    
#     return Q.value, Y.value, sigma.value

def LMI_v2(A, B,  alpha, verbose=False):
    xdim=3
    Q = cvx.Variable((xdim,xdim), symmetric=True)
    Y = cvx.Variable((1,xdim))
    F_d = cvx.Variable()
    F_w = cvx.hstack((0,F_d))
    BFw = cvx.vstack((cvx.hstack((-1, 0)), F_w, cvx.hstack((0, 1))))
    F_w = cvx.vstack(cvx.hstack((0,F_d))).T
    # cvx_Bw = Bw + BFw# Bw + BFw
    gamma = cvx.Variable(nonneg=True)
    sigma = cvx.Variable(nonneg=True) # works at 1750 with F_d, 2500 without F_d
    LMI_1 = cvx.vstack((
        cvx.hstack((-Q + alpha*Q, 0*np.ones((xdim,2)), Q@A.T+Y.T@B.T)), #,
        cvx.hstack((0*np.ones((2,xdim)), -alpha*np.eye(2,2), BFw.T)),
        cvx.hstack((A@Q + B@Y, BFw, -Q))))

    LMI_2 = cvx.vstack((
        cvx.hstack((Q, 0*np.ones((xdim,2)), Y.T)), # 
        cvx.hstack((0*np.ones((2,xdim)),(gamma- sigma)*np.eye(2), F_w.T)),
        cvx.hstack((Y, F_w, cvx.vstack((sigma,))))        # 
        ))
    constraints = [Q>>0, LMI_1<<0, LMI_2>>0] # 
    find_f = cvx.Problem(cvx.Minimize(gamma), constraints=constraints)
    try:
        find_f.solve(solver=cvx.MOSEK,verbose=verbose) #
        # print(f'alpha is {alpha.value}')
    except cvx.error.SolverError  as e:
        print(f' solver cant solve alpha = {alpha}')
        return 0, 0, 0, 0, 0
    return Q.value, Y.value, gamma.value, sigma.value, F_d.value

def LMI_noforecast(A, B,  alpha, verbose=False):
    A_nof = A[:2, :2]
    B_nof = B[:2]
    xdim=2
    Q = cvx.Variable((xdim,xdim), symmetric=True)
    Y = cvx.Variable((1,2))
    Bw = cvx.vstack((cvx.hstack((-1, -1)), cvx.hstack((0, 0))))

    gamma = cvx.Variable(nonneg=True)
    sigma = cvx.Variable(nonneg=True) # works at 1750 with F_d, 2500 without F_d
    LMI_1 = cvx.vstack((
        cvx.hstack(((alpha - 1 )*Q, 0*np.ones((xdim,2)), Q@A_nof.T+Y.T@B_nof.T)), #,
        cvx.hstack((0*np.ones((2,xdim)), -alpha*np.eye(2,2), Bw.T)),
        cvx.hstack((A_nof@Q + B_nof@Y, Bw, -Q))))

    LMI_2 = cvx.vstack((
        cvx.hstack((Q, 0*np.ones((xdim,1)), Y.T)), # 
        cvx.hstack((0*np.ones((1,2)),(gamma- sigma)*np.eye(1), 0*np.ones((1, 1)))),
        cvx.hstack((Y, 0*np.ones((1,1)), cvx.vstack((sigma,))))        # 
        ))
    constraints = [Q>>0, LMI_1<<0, LMI_2>>0] # 
    find_f = cvx.Problem(cvx.Minimize(gamma), constraints=constraints)
    try:
        find_f.solve(solver=cvx.MOSEK,verbose=verbose) #

    except cvx.error.SolverError  as e:
        print(f' solver cant solve alpha = {alpha}')
        return 0, 0, 0, 0, 0
    return Q.value, Y.value, gamma.value, sigma.value, 0
    
# noises = 0.1*np.random.rand(1000) - 1.
# # noises = -np.ones(1000)

# #----------- H2 gain solver ---------------#
# P, F2, F_d2 = LMI_H2(A_f, B, C, Bw)
# F_x = F2.dot(np.linalg.inv(P))# feedback =FP^{-1}x + F_d d
# #-----------------end H2 ------------------#
# hist_h2 = sim(A_f + B.dot(F_x), Bw, noises, F_d=F_d2)
# plot(hist_h2, C, 'H2 gain')
# #----------- peak gain solver ---------------#
# # solving it for one alpha
# Q, Y, sigma, F_dp = LMI_book(A_f, B, Bw, C, D, 0.9)
# F_x = Y.dot(np.linalg.inv(Q))# feedback = YQ^{-1}x + F_wd
# #--------------- end peak gain --------------#
# hist_peak = sim(A_f + B.dot(F_x), Bw, noises, F_d=F_dp)
# plot(hist_peak, C, 'peak gain')
# #------------ end peak gain sim --------------#
# hist = sim(A_f, D, noises, F_d=30)
# plot(hist)

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
def compute_gam(A, B, alpha, lamb_list, gam_list, func, T):
    Q, Y, gamma, sigma, F_d = func(A,B,(alpha+1)/T)
    if gamma == 0 and sigma == 0 or gamma is None:
        return
    else:
        # print(f'found gamma value {gamma}')
        gam_list.append(gamma)
        lamb_list.append(alpha/T)
    return 
# Q, Y, x, sigma, F_d = LMI_v2(A,B, 0.01)
# F = Y.dot(np.linalg.inv(Q))
# F_d = F_d/np.sqrt(sigma)
# A_c = A + B.dot(F)
# Bw = np.array([[1, 0], [0, F_d], [0, 1]])
# --------- varying backlog rate --------------- # 
params = [(0.5, 0.1), (0.5, 0.3), (0.5, 0.5), (0.5, 0.7), (0.5, 0.9)] # , (0.01, 0.01), (0.2, 0.2), (0.1, 0.9)
gammas = []
gammas_no_forecast = []

lambs = []
lambs_no_forecast = []
for (beta, backlog) in params:
    A = np.array([[1-beta, 1-backlog, -1], [0, backlog, 0], [0, 0, 0]])
    B = np.array([[0],[1], [0]])
    T = 100
    min_gam = []
    min_gam_no_forecast = []
    lamb = []
    lamb_no_forecast = []
    F_d = []
    for alpha in range(T-1):
        print(f'\r {alpha} ', end = '')
        compute_gam(A, B, alpha, lamb, min_gam, LMI_v2, T)
        # compute_gam(A, B, alpha, lamb_no_forecast, min_gam_no_forecast, 
        #             LMI_noforecast)
        

    gammas.append(min_gam)
    lambs.append(lamb)
    gammas_no_forecast.append(min_gam_no_forecast)
    lambs_no_forecast.append(lamb_no_forecast)
    
plt.figure()
plt.title(fr'Peak gains $\gamma$ at perishable rate $\beta$={beta}')
for i in range(len(params)):
    plt.plot(lambs[i], [np.sqrt(g) for g in gammas[i]], label=fr'$\alpha$={params[i][1]}', color= f'C{i}',linewidth=2); 
    # plt.plot(lambs_no_forecast[i], [np.sqrt(g) for g in gammas_no_forecast[i]], 
    #           label=f'{params[i]}',linestyle='dashdot', color= f'C{i}'); 
plt.yscale('log'); plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\gamma$'); plt.legend(); plt.grid(); 
plt.show(block=False)

# --------- varying perishing rate --------------- # 
params = [(0.1, 0.5), ( 0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.9, 0.5)] # , (0.01, 0.01), (0.2, 0.2), (0.1, 0.9)
gammas = []
gammas_no_forecast = []
lambs = []
lambs_no_forecast = []
for (beta, backlog) in params:
    A = np.array([[1-beta, 1-backlog, -1], [0, backlog, 0], [0, 0, 0]])
    B = np.array([[0],[1], [0]])
    T = 100
    min_gam = []
    min_gam_no_forecast = []
    lamb = []
    lamb_no_forecast = []
    for alpha in range(T-1):
        print(f'\r {alpha} ', end = '')
        compute_gam(A, B, alpha, lamb, min_gam, LMI_v2, T)
        # compute_gam(A, B, alpha, lamb_no_forecast, min_gam_no_forecast, 
        #             LMI_noforecast)

    gammas.append(min_gam)
    lambs.append(lamb)
    gammas_no_forecast.append(min_gam_no_forecast)
    lambs_no_forecast.append(lamb_no_forecast)
plt.figure()
plt.title(fr'Peak gains $\gamma$ at backlog rate $\alpha$={backlog}')
for i in range(len(params)):
    plt.plot(lambs[i], [np.sqrt(g) for g in gammas[i]], label=fr'$\beta$={params[i][0]}', color= f'C{i}',linewidth=2); 
    # plt.plot(lambs_no_forecast[i], [np.sqrt(g) for g in gammas_no_forecast[i]], 
    #           label=f'{params[i]}',linestyle='dashdot', color= f'C{i}'); 
plt.yscale('log'); plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\gamma$'); plt.legend(); plt.grid(); 
plt.show(block=False)

# ------------- controller properties -------------- 
# backlog= 0.1
# beta = 0.1
# A = np.array([[1-beta, 1-backlog, -1], [0, backlog, 0], [0, 0, 0]])
# B = np.array([[0],[1], [0]])
# Q, Y, x, sigma, F_d = LMI_v2(A,B, 0.01)
# F = Y.dot(np.linalg.inv(Q))
# F_d = F_d/np.sqrt(sigma)
# A_c = A + B.dot(F)
# Bw = np.array([[1, 0], [0, F_d], [0, 1]])


# u_hists = []
# i_hists = []
# noise_thresh = [1000, 500, 10]
# eps_d = 1000
# eps_sample = 1000
# ef_sample = []
# for ef_ind in range(eps_sample):
#     noises = eps_d*np.random.rand(1000,2) - eps_d
#     ef =  0.5*ef_ind/eps_sample*eps_d
#     noises[:,0] = ef*noises[:,0] 
#     ef_sample.append(ef)
#     hist = sim(A_c, B, Bw, noises)
#     u_hist = [np.squeeze(F).dot(x_k)+ F_d*n_k[1] for x_k, n_k in zip(hist, noises) ]
    
    
#     i_hist = [x_k[0] for x_k in hist]
#     u_hists.append(max([abs(u) for u in u_hist]) )
#     i_hists.append(max([abs(u) for u in i_hist]) )

# plt.figure()
# plt.title(r'Empirical order fluctuations value at $\epsilon_d = 1000$')
# plt.plot(ef_sample, u_hists, label=r'$\max_k |o(k) - o^\infty|$')
# plt.plot(ef_sample, [np.sqrt(x*(ef**2 + (eps_d)**2)) for ef in ef_sample], label=r'bound$')  
# plt.grid(); plt.legend();plt.yscale('log');plt.xlabel(r'$\epsilon_f$')
# plt.show(block=False)

# plt.figure()
# plt.title(r'Empirical inventory fluctuations at $\epsilon_d = 1000$')
# # plt.plot(ef_sample, u_hists, label=r'$o(k) - o^\infty$')
# plt.plot(ef_sample, i_hists, label=r'$\max_k |i(k) - i^\infty$')  
# plt.grid(); plt.legend();plt.yscale('log');plt.xlabel(r'$\epsilon_f$')
# plt.show(block=False)
# ------------- controller properties -------------- 
    
# for ef in noise_thresh:
#     noises = 2*eps_d*np.random.rand(1000,2) - eps_d
#     noises[:, 0] = ef*noises[:, 0]
#     hists.append(sim(A_c, B, Bw, noises))

# cols = ['order', 'inventory']
# plt.figure()
# plt.title(r'Order fluctuations $o(k) - o^\infty$')
# for i, hist in enumerate(hists):
#     obs_hist = [np.squeeze(F).dot(x_k)[0]+ F_d*n_k[1] for x_k, n_k in zip(hist, noises) ]# 
#     plt.plot(obs_hist, alpha=0.5, label=fr'$\epsilon_f/\epsilon_d =$ {noise_thresh[i]/eps_d}')
# plt.ylabel(r'$| o(k) - o^\infty|$')
# plt.grid(); plt.legend();plt.yscale('log');plt.xlabel('Time step (k)')
# plt.show(block=False)


# cols = ['order', 'inventory']
# plt.figure()
# plt.title(r'Inventory fluctuations $i(k) - i^\infty$')
# for i, hist in enumerate(hists):
#     obs_hist = [x_k[0] for x_k in hist ]# 
#     plt.plot(obs_hist, alpha=0.5, label=fr'$\epsilon_f/\epsilon_d =$ {noise_thresh[i]/eps_d}')

# plt.grid(); plt.legend(); plt.yscale('log');plt.xlabel('Time step (k)')
# plt.ylabel(r'$| i(k) - i^\infty|$')
# plt.show(block=False)



# Q_f, Y_f, x_f, sigma_f, F_df = LMI_v2(A,B,0.02)
# F_f = Y_f.dot(np.linalg.inv(Q_f))
# F_df = F_df/np.sqrt(sigma_f)
# A_cf = A + B.dot(F_f)
# Bwf = np.array([[1, 0], [0, F_df], [0, 1]])

# hist = sim(A_cf, B, Bwf, noises)
# plot(hist, np.squeeze(F_f))

# hist = sim(A, B, D, noises, F_d=30)
# plot(hist)
# backlog= 0.1
# beta = 0.9
# A = np.array([[1, backlog], [0, 1-backlog]])
# B = np.array([[0],[1]])
# Bw = np.array([[-1, 0], [0, 0]])
# T = 100
# for has_forecast in [True, False]:
#     min_sigma = []
#     for alpha in range(T-1):
#         print(f'\r {alpha} ', end = '')
#         Q, Y, sigma = LMI_v2(A,B,Bw,(alpha+1)/T,perfect_forecast=has_forecast)
#         min_sigma.append(sigma)
#     plt.plot(min_sigma, label=f'has perfect forecast {has_forecast}')
# plt.legend()
# plt.yscale('log')
# plt.grid()
# plt.show(block=False)
# # LTI system
# for trial in range(1):
#     backlog = np.random.rand() # backlog = alph
#     beta = np.random.rand() 
#     backlog= 0.1
#     beta = 0.9
#     if beta < backlog:
#         backlog_tmp = beta
#         beta =backlog
#         backlog = backlog_tmp
        
#     A = np.array([[1, backlog], [0, 1-backlog]])
#     B = np.array([[0],[1]])
    
#     Bw = np.array([[-1, 0], [0, 0]])
    
#     plt.figure()
#     plt.title(f'Backlog={1-backlog}, perish rate = {1-beta}')
#     for eps_f in [0,10]:
        
#         T = 100
#         min_sigma = []
#         for alpha in range(T-1):
#             print(f'\r {alpha} ', end = '')
#             Q, Y, sigma, F_d = LMI_book(A,B,Bw,(alpha+1)/T, eps_f = eps_f)
#             min_sigma.append(sigma)
#         plt.plot(min_sigma, label=f'{eps_f}')
#     plt.legend()
#     plt.yscale('log')
#     plt.grid()
#     plt.show(block=False)
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






