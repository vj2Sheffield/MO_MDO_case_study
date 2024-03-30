# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:08:38 2023

@author: Victoria Johnson
"""
import numpy as np
from scipy.optimize import minimize as minimizeScipy
from scipy import optimize
import matplotlib.pyplot as plt
global i
i = 0

def sub1(vy2, u, B, intercept):
    z1_out = intercept[0] + B[0][2]*u[1] + B[0][1]*vy2[1] + B[0][0]*vy2[0]
    return z1_out

def sub2(vy1, u, A, B, intercept, u2_baseline):
    z3_out = intercept[2] + B[2][0]*u[0]
    z4_out = intercept[3] + B[3][0]*u[0] + B[3][1]*u2_baseline
    z2_out = intercept[1] + A[1][0]*vy1 + A[1][2]*z3_out + A[1][3]*z4_out 
    z5_out = intercept[4] + B[4][0]*u[0] + A[4][1]*z2_out
    z6_out = intercept[5] + B[5][0]*u[0] + A[5][0]*vy1 + A[5][1]*z2_out + A[5][4]*z5_out + B[5][1]*u2_baseline
    z7_out = intercept[6] + A[6][5]*z6_out
    z8_out = intercept[7] + A[7][5]*z6_out
    return np.array((u[0], u2_baseline, z6_out, z8_out, z2_out, z3_out, z4_out, z5_out, z7_out))

def solver(u, A, B, intercept, u2_baseline):
    N = 200
    global i
    myEps = 1e-8
    yout1_prev = 0
    yout2_prev = 0
    yout1 = optimize.newton(sub1, np.ones(2), args = (u, B, intercept), full_output = True)
    yout1_fun = sub1(yout1, u, B, intercept)
    yout2 = sub2(yout1_fun, u, A, B, intercept, u2_baseline)
    
    for m in range(0, N):
        yout1 = sub1(yout2, u, B, intercept)
        yout2 = sub2(yout1, u, A, B, intercept, u2_baseline)
        
        if m > 1 and abs(yout1 - yout1_prev) < myEps and abs(sum(yout2) - sum(yout2_prev)) < myEps:
            # print('Solved in ' + str(m) + ' iterations.')
            break
        yout1_prev = yout1
        yout2_prev = yout2
        if m == N - 1:
            raise Exception('Solver has not converged in time.')
        i = i + 1
        # print(i)
    return yout1, yout2

def SII(HLE, POV): # Calculate slope index of inequality     
    # POV_rank = np.flip(np.argsort(POV))
    POV_rank = np.argsort(POV)
    POV_rank = HLE[POV_rank]
    
    POV = np.vstack([np.array(range(0,len(POV))), np.ones(len(POV))]).T     # Add 'x' data for regressio
    
    m,c = np.linalg.lstsq(POV, POV_rank, rcond=None)[0]   # Do linear regression
    # print('SII = ' + str(m*10 - m*0))   
    return m*10 - m*0   # Return SII

# xticklabels = ['TAM', 'OLD', 'ROC', 'BOL', 'MAN', 'SAL', 'WIG', 'BUR', 'STO', 'TRA']
xticklabels = ['BOL', 'BUR', 'MAN', 'OLD', 'ROC', 'SAL', 'STO', 'TAM', 'TRA', 'WIG']

def plot_SII(HLE, POV, pops): # Calculate slope index of inequality 
    # POV_rank = np.flip(np.argsort(POV))
    POV_rank = np.argsort(POV)
    ordered_x_ticks = [xticklabels[i] for i in POV_rank]
    POV_rank = HLE[POV_rank]  
    
    POV_new = np.vstack([np.array(range(0,len(POV))), np.ones(len(POV))]).T     # Add 'x' data for regressio
    
    m,c = np.linalg.lstsq(POV_new, POV_rank, rcond=None)[0]   # Do linear regression
    
    print('SII = ' + str(m*10 - m*0))   
    print('WM MCS = ' + str(round(sum(HLE*pops)/sum(pops), 4)))
    SII_str = 'SII = ' + str(round(m*10 - m*0, 4))
    avg_HLE_str = 'WM MCS = ' + str(round(sum(HLE*pops)/sum(pops), 4))
    # avg_HLE_str = 'WM HLE = ' + str(round(np.mean(HLE), 4))
    plt.scatter(range(0, 10), POV_rank)
    plt.plot([0, 9], [m*0 + c, m*10 + c], 'r--')
    plt.ylim([45, 55])
    plt.xlabel("EPR-ranked LAs")
    plt.ylabel("MCS")
    plt.text(0, 54, SII_str)
    plt.text(0, 53, avg_HLE_str)
    # plt.title('Best case scenario')
    
    plt.xticks(np.arange(0, 10), ordered_x_ticks)
    plt.show()
    return(m*10 - m*0)

def unpack_coefs(df):
    intercept = df[:, 1]
    A_mat = df[:, 5:13]
    B_mat = df[:, 2:5]
    return intercept, A_mat, B_mat

def inv_transforms_z(u, mu, sigma, z_no):
    u_denorm = u*sigma + mu
    if z_no == 1 or z_no == 2 or z_no == 3:
        return 100*np.exp(u_denorm)/(1 + np.exp(u_denorm))
    elif z_no == 4 or z_no == 8:
        return np.exp(u_denorm)
    elif z_no == 5:
        return np.exp(u_denorm) + 1
    elif z_no == 6:
        return 12 + (48*np.exp(u_denorm))/(1 + np.exp(u_denorm))
    elif z_no == 7:
        return 100000*np.exp(u_denorm)/(1 + np.exp(u_denorm))
        
def nonlinear_transforms_u(u, mu, sigma):
    if u <= 0:
        u = 1e-15
    if u >= 100:
        u = 99.9999
    u_trans = np.log(u/(100 - u))
    u_norm = (u_trans - mu)/sigma
    return u_norm

def x_to_u(x, beta, gamma, mu, sigma):
    if np.ravel(x).shape[0] > 22:
        u = np.zeros((x.shape[0], 20))
        for m in range(0, 10):
            u[:, 2*m] = nonlinear_transforms_u((x[:, 2*m]*beta[m, 0] + gamma[m, 0]), mu[0], sigma[0])    # Labour remuneration
            u[:, 2*m + 1] = nonlinear_transforms_u((x[:, 2*m + 1]*beta[m, 2] + gamma[m, 2]), mu[2], sigma[2])    # Skills and qualifications
   
    else:
        u = np.zeros((20))
        for m in range(0, 10):
            u[2*m] = nonlinear_transforms_u((x[2*m]*beta[m, 0] + gamma[m, 0]), mu[0], sigma[0])    # Labour remuneration
            u[2*m + 1] = nonlinear_transforms_u((x[2*m + 1]*beta[m, 2] + gamma[m, 2]), mu[2], sigma[2])    # Skills and qualifications
   
    return u

def x_to_u_no_norm(x, beta, gamma):
    u = np.zeros((30))
    for m in range(0, 10):
        u[2*m] = x[2*m]*beta[m, 0] + gamma[m, 0]    # Labour remuneration
        u[2*m + 1] = x[2*m + 1]*beta[m, 2] + gamma[m, 2]    # Skills and qualifications
        
    return u
    
def nonlinear_transforms_z(u, mu, sigma, z_no):
    if z_no == 1 or z_no == 2 or z_no == 3:
        u_norm = np.log(u/(100 - u))
    if z_no == 4 or z_no == 8:
        u_norm = np.log(u)
    if z_no == 5:
        u_norm = np.log(u - 1)
    if z_no == 6:
        u_norm = np.log((u - 12)/(60 - u))
    if z_no == 7:
        u_norm = np.log(u/(100000 - u))
    
    u_trans = (u_norm - mu)/sigma
    return u_trans


def full_model(u, A, B, intercept):
    z1_out = intercept[0] + B[0][2]*u[2]+ B[0][1]*u[1] + B[0][0]*u[0]
    z3_out = intercept[2] + B[2][0]*u[0]
    z4_out = intercept[3] + B[3][0]*u[0] + B[3][1]*u[1]
    z2_out = intercept[1] + A[1][0]*z1_out + A[1][2]*z3_out + A[1][3]*z4_out 
    z5_out = intercept[4] + B[4][0]*u[0] + A[4][1]*z2_out
    z6_out = intercept[5] + B[5][0]*u[0] + B[5][1]*u[1] + A[5][0]*z1_out + A[5][1]*z2_out + A[5][4]*z5_out
    z7_out = intercept[6] + A[6][5]*z6_out
    z8_out = intercept[7] + A[7][5]*z6_out
    return np.array((z1_out, z2_out, z3_out, z4_out, z5_out, z6_out, z7_out, z8_out))



def full_model_with_JPC(u, A, B, intercept, u2_baseline):
    z1_out = intercept[0] + B[0][2]*u[1]+ B[0][1]*u2_baseline + B[0][0]*u[0]
    z3_out = intercept[2] + B[2][0]*u[0]
    z4_out = intercept[3] + B[3][0]*u[0] + B[3][1]*u2_baseline
    z2_out = intercept[1] + A[1][0]*z1_out + A[1][2]*z3_out + A[1][3]*z4_out 
    z5_out = intercept[4] + B[4][0]*u[0] + A[4][1]*z2_out
    z6_out = intercept[5] + B[5][0]*u[0] + B[5][1]*u2_baseline + A[5][0]*z1_out + A[5][1]*z2_out + A[5][4]*z5_out
    z7_out = intercept[6] + A[6][5]*z6_out
    z8_out = intercept[7] + A[7][5]*z6_out
    return np.array((z1_out, z2_out, z3_out, z4_out, z5_out, z6_out, z7_out, z8_out))







