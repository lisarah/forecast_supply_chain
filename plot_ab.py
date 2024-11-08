#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 00:31:32 2024

@author: huilih
"""
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import itertools as it

def LMI(A, B,  alpha, verbose=False):
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


noises = 0.1*np.random.rand(1000) - 1.

backlog = []
T = 100
alpha = [p*0.01 for p in range(T)]
beta = [0.1] #, 0.5, 0.9
for b in beta:
    min_sigma = []
    lambdas = []
    backlogs = []
    for lambda_, a in it.product([p*0.01 for p in range(T)], repeat=2):
        print(f'\r  beta = {b} ----    on point lambda = {lambda_}, backlog = {a}, ----   ', end='')
        A = np.array([[1-b, 1-a, -1], [0, a, 0], [0, 0, 0]])
        B = np.array([[0],[1], [0]])
        Q, Y, gamma, sigma, F_d = LMI(A,B,(lambda_+1)/T)
        if gamma == 0 and sigma == 0 or gamma is None:
            continue
        else:
            min_sigma.append(gamma)
            lambdas.append(lambda_)
            backlogs.append(a)
    plt.figure()
    plt.title(fr'$ \beta$ = {b}')
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid([p*0.01 for p in range(1, T+1)], [p*0.01 for p in range(1, T+1)])

    Z = []
    for i_ind in range(100):
        Z.append([])
        for j_ind in range(100):
            Z[-1].append(min_sigma[j_ind + i_ind*100])

        
    # Data for a three-dimensional line
    ax.contour3D(X, Y, Z, 50, cmap='bone')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\alpha$')
    ax.set_zlabel(r'$\gamma$');
    plt.show(block=False)

# noises = 2*np.random.rand(1000,2) - 1
# noises[:, 0] = 0.5*noises[:, 0]


# Q, Y, x, sigma, F_d = LMI_noforecast(A,B,0.02)
# F = Y.dot(np.linalg.inv(Q))
# F_d = F_d/np.sqrt(sigma)
# A_c = A + np.hstack((B.dot(F), np.zeros((3,1))))
# Bw = np.array([[1, 0], [0, F_d], [0, 1]])
# hist = sim(A_c, B, Bw, noises)
# plot(hist, np.hstack((np.squeeze(F), 0)))

# Q_f, Y_f, x_f, sigma_f, F_df = LMI_v2(A,B,0.02)
# F_f = Y_f.dot(np.linalg.inv(Q_f))
# F_df = F_df/np.sqrt(sigma_f)
# A_cf = A + B.dot(F_f)
# Bwf = np.array([[1, 0], [0, F_df], [0, 1]])

# hist = sim(A_cf, B, Bwf, noises)
# plot(hist, np.squeeze(F_f))




