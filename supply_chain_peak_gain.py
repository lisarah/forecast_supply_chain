# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

# LTI system
backlog = 0.1 # backlog = alpha
A = np.array([[1, backlog], [0, 1-backlog]])
B = np.array([[0],[1]])
# A = np.random.rand(2,2)
# B = np.random.rand(2,1)

Bw = np.array([[-1], [0]])
Dz = np.array([[1], [0]])

def sim(A, Bw, noises, F_d = None, init_state=None):
    
    if init_state is None:
        init_state = np.random.rand(2,1)
        # print(f' initial state is {init_state}')
    if F_d is not None:
        BwF = Bw + F_d
    
    hist = [init_state]
    for d_k in noises:
        x_k = hist[-1]
        hist.append(A.dot(x_k) + BwF.dot(d_k))
        # print('   another noise    ')
        # print(f'DF_d d = {D_f.dot(d_k)}')
        # print(f'(A+BF)x = {A.dot(x_k)}')
        # print(A.dot(x_k) + D_f.dot(d_k))
        # print('--------end ------------')
    return hist

def plot(hist, C, title=None):
    cols = ['inventory', 'pipeline']
    plt.figure()
    obs_hist = [C.dot(x_k) for x_k in hist ]
    if title is not None:
        plt.title(title)
    for i in range(len(obs_hist[0])):
        plt.plot([z_k[i] for z_k in obs_hist], label=cols[i],alpha=0.5)
    plt.grid(); plt.legend();
    plt.show(block=False)
    

def LMI_Fd(A, F_p, alpha):
    P = cvx.Variable((2,2), symmetric=True)
    F_d = cvx.Variable()
    beta = cvx.Variable(nonneg=True)
    gamma = cvx.Variable(nonneg=True)
    Ainv = np.linalg.pinv(A)
    Bw = cvx.vstack((1, F_d))
    ABw = Ainv@Bw
    gamma2 = gamma*np.eye(2)
    beta1 = beta*np.eye(1)
    LMI_1 = cvx.vstack((cvx.hstack((P, ABw)), cvx.hstack((ABw.T, -beta1))))
    LMI_2 = cvx.vstack((cvx.hstack((P, gamma2)), cvx.hstack((gamma2, np.eye(2)))))
    constraints = [P>>0]
    constraints.append(LMI_1 >>0)
    constraints.append(LMI_2 << 0.001*np.eye(4))
    constraints.append(F_d >= 1 + F_p/alpha)
    
    find_fd = cvx.Problem(cvx.Minimize(gamma), constraints)
    find_fd.solve()
    return P.value, F_d.value

def LMI_book(A, B, Bw, C, D, alpha):
    c_dim, _ = C.shape
    Q = cvx.Variable((2,2), symmetric=True)
    # alpha = 0.9# 0.8#cvx.Variable(nonneg=True)
    # print(f'alpha is {alpha}')
    Y = cvx.Variable((1,2))
    F_d = cvx.Variable()
    # F_d = 0
    BFw = cvx.vstack((0, F_d))
    cvx_Bw = Bw + BFw# Bw + BFw
    
    sigma = cvx.Variable(nonneg=True) # works at 1750 with F_d, 2500 without F_d
    LMI_1 = cvx.vstack((
        cvx.hstack((-Q + alpha*Q, 0*np.ones((2,1)), Q@A.T+Y.T@B.T)), #,
        cvx.hstack((0*np.ones((1,2)), -alpha*np.ones((1,1)), cvx_Bw.T)),
        cvx.hstack((A@Q + B@Y, cvx_Bw, -Q))))

    LMI_2 = cvx.vstack((
        cvx.hstack((-Q, Q@C.T+ Y.T@D.T)), # 
        cvx.hstack((C@Q+ D@Y , -sigma*np.eye(c_dim))) # 
        ))
    constraints = [Q>>0, LMI_1<<0, LMI_2<<0] # 
    find_f = cvx.Problem(cvx.Minimize(sigma), constraints=constraints)
    find_f.solve(solver=cvx.MOSEK) #,verbose=True)
    
    return Q.value, Y.value, sigma.value, BFw.value

def LMI_H2(A, B, C, Bw):
    P = cvx.Variable((2,2), symmetric=True)
    z_dim, _ = C.shape
    Z = cvx.Variable((z_dim,z_dim), symmetric=True)
    F = cvx.Variable((1,2))
    F_d = cvx.Variable()
    BFw = cvx.vstack((0, F_d))
    cvx_Bw = Bw + BFw# Bw + BFw
    # print(B.shape)
    # print(F.shape)
    # print((B@F).shape)
    # print((A@P).shape)
    # print(Bw.T.shape)
    LMI_r1 = cvx.hstack((P, A@P+ B@F, cvx_Bw))
    LMI_r2 = cvx.hstack((P@A.T + F.T@B.T, P, 0*np.ones((2,1))))
    LMI_r3 = cvx.hstack((cvx_Bw.T, 0*np.ones((1,2)), np.ones((1,1))))
    # print(LMI_r1.shape)
    # print(LMI_r2.shape)
    # print(LMI_r3.shape)
    LMI_1 = cvx.vstack((LMI_r1, LMI_r2, LMI_r3))
    LMI_2r1 = cvx.hstack((Z, C@P))
    LMI_2r2 = cvx.hstack((P@C.T, P))
    LMI_2 = cvx.vstack((LMI_2r1, LMI_2r2))
    constraints = [LMI_1 >> 0, LMI_2 >>0, P >> 0, Z>>0]
    find_fd = cvx.Problem(cvx.Minimize(cvx.trace(Z)), constraints)
    find_fd.solve(solver='MOSEK')
    return P.value, F.value, BFw.value
    
# def LMI_Q(A, F_p, alpha):
#     Q = cvx.Variable((2,2), symmetric=True)
#     F_d = backlog*(backlog + F_p) + 10
#     alpha = 0.1
#     Bw = cvx.vstack((1, F_d))
#     LMI = 1/alpha * Bw@Bw.T  - Q - 1/(1-alpha)*A_f@Q@A_f.T
#     constraints = [LMI<<0, Q>>0]
#     find_fd = cvx.Problem(cvx.Minimize(cvx.tr_inv(Q)), constraints)
#     find_fd.solve(solver=cvx.MOSEK)
#     return Q.value, F_d.value

def grad_est(alpha, batch_num, A, B, C, D, Bw, var=0.3, eps=1e-5):
    delta_f = []
    Q, Y, sigma, F_dp = LMI_book(A, B, Bw, C, D, alpha)
    for b in range(batch_num):
        mu_k = var*2*(np.random.random_sample() - 0.5)
        # print(f'muk is {mu_k}')
        u_k = np.sign(mu_k)
        alpha_k = max(min(alpha+mu_k, 1), eps)
        Q_k, Y_k, sigma_k, F_dp_k = LMI_book(A, B, Bw, C, D, alpha_k)
        delta_f.append((sigma_k - sigma)/mu_k*u_k)
    return sum(delta_f)/batch_num

    
noises = 0.1*np.random.rand(1000) - 1.
# noises = -np.ones(1000)
conv = 1.1
F_p = conv - backlog
F_i = conv**2/4/backlog
BF = np.array([[0,0], [-F_i, -F_p]])
A_f = A + BF
D = np.zeros((1,1))
C = np.array([[1, 0]])

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
T = 300
min_sigma = []
for alpha in range(T):
    print(f'\r {alpha} ', end = '')
    Q, Y, sigma, F_d = LMI_book(A,B,Bw,Dz, (alpha+1)/T)
    min_sigma.append(sigma)
# Y.dot(np.linalg.inv(Q))
plt.figure()
plt.plot(min_sigma)
plt.grid()
plt.show(block=False)
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






