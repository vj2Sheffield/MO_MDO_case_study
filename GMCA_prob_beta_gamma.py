# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:52:40 2023

@author: Victoria Johnson
"""

from GMCA_functions_beta_gamma import solver, SII, unpack_coefs, inv_transforms_z, x_to_u, plot_SII, full_model, nonlinear_transforms_z

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2

save_results = True

pops = np.array(pd.read_csv('Model_Data/populations.csv', header = None))[:, 1]
xls = pd.ExcelFile('Model_Data/Static_Model_Parameters.xlsx')
df2 = np.array(pd.read_excel(xls, 'Cluster2', index_col = 0))
df3 = np.array(pd.read_excel(xls, 'Cluster3', index_col = 0))
df4 = np.array(pd.read_excel(xls, 'Cluster4', index_col = 0))
xticklabels = ['BOL', 'BUR', 'MAN', 'OLD', 'ROC', 'SAL', 'STO', 'TAM', 'TRA', 'WIG']
beta = np.array(pd.read_csv('Model_Data/beta.csv', header = None))
gamma = np.array(pd.read_csv('Model_Data/gamma.csv', header = None))


residuals = df4[:, 0]
G1 = 30.382003
N = 22
lb = np.zeros(N)
ub = G1*np.ones(N)
myEps = 1
gLim = 99

PVT = np.zeros((10))
EPR = np.zeros((10))
FPR = np.zeros((10))
DHI = np.zeros((10))
EIQ = np.zeros((10))
MCS = np.zeros((10))
PMR = np.zeros((10))
HLE = np.zeros((10))

Cl4_intercept, Cl4_A, Cl4_B = unpack_coefs(df4)
Cl2_intercept, Cl2_A, Cl2_B = unpack_coefs(df2)
Cl3_intercept, Cl3_A, Cl3_B = unpack_coefs(df3)

cluster_A = [Cl4_A, Cl3_A, Cl4_A, Cl4_A, Cl4_A, Cl4_A, Cl3_A, Cl4_A, Cl2_A, Cl3_A]
cluster_B = [Cl4_B, Cl3_B, Cl4_B, Cl4_B, Cl4_B, Cl4_B, Cl3_B, Cl4_B, Cl2_B, Cl3_B]
cluster_int = [Cl4_intercept, Cl3_intercept, Cl4_intercept, Cl4_intercept, Cl4_intercept, Cl4_intercept, Cl3_intercept, Cl4_intercept, Cl2_intercept, Cl3_intercept]


mu = [-1.147, -3.0625, 0.9039, 1.0756, -0.9247, -2.1, 9.7798, 0.2081, 1.2153, -5.65659, 4.1368]
sigma = [0.3060, 0.4554, 0.4370, 0.2981, 0.2773, 0.1388, 0.2701, 0.1388, 0.2057, 0.2064, 0.0538]
baseline_JPC = np.array([3.3, 4.8, 6.4, 3.9, 3.7, 4.5, 6.1, 5.6, 5.5, 4])

t0 = time.time()
class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=N,
                          n_obj=3,
                          n_ieq_constr = 7,
                          n_eq_constr = 0,
                          xl = lb,
                          xu = ub)

    def _evaluate(self, x, out, *args, **kwargs): 
        max_g = np.zeros(20)
        # 1. Get u using x, beta and gamma. Transform it into 'model' units
        u = x_to_u(x, beta, gamma, mu, sigma)
        
        for n in range(0, 10):
            LA1_yout1, LA1_yout2 = solver(u[n*2:n*2 + 2], cluster_A[n], cluster_B[n], cluster_int[n], baseline_JPC[n])
            EPR[n] = inv_transforms_z(LA1_yout1, mu[3], sigma[3], 1)   # z1
            PVT[n] = inv_transforms_z(LA1_yout2[4], mu[4], sigma[4], 2) # z2
            FPR[n] = inv_transforms_z(LA1_yout2[5], mu[5], sigma[5], 3) # z3
            DHI[n] = inv_transforms_z(LA1_yout2[6], mu[6], sigma[6], 4) # z4
            EIQ[n] = inv_transforms_z(LA1_yout2[7], mu[7], sigma[7], 5) # z5
            MCS[n] = inv_transforms_z(LA1_yout2[2], mu[8], sigma[8], 6) # z6
            PMR[n] = inv_transforms_z(LA1_yout2[8], mu[9], sigma[9], 7) # z7
            HLE[n] = inv_transforms_z(LA1_yout2[3], mu[10], sigma[10], 8) # z8
            
        HLE_weighted_sum = sum(MCS*pops)/sum(pops)
        
        g1 = (x[20] + x[21]) - G1
        g2 = abs(x[0] + x[2] + x[4] + x[6] + x[8] + x[10] + x[12] + x[14] + x[16] + x[18] - x[20]) - myEps
        g3 = abs(x[1] + x[3] + x[5] + x[7] + x[9] + x[11] + x[13] + x[15] + x[17] + x[19] - x[21]) - myEps
        for m in range(0, 10):
            max_g[2*m] = inv_transforms_z(u[2*m], mu[0], sigma[0], 1)
            max_g[2*m + 1] = inv_transforms_z(u[2*m + 1], mu[0], sigma[0], 1)
        g4 = max(max_g) - gLim
        g5 = 1 - min(max_g)
        g6 = x[0] + x[2] + x[4] + x[6] + x[8] + x[10] + x[12] + x[14] + x[16] + x[18] - x[20]
        g7 = x[1] + x[3] + x[5] + x[7] + x[9] + x[11] + x[13] + x[15] + x[17] + x[19] - x[21]
        
        f1 = -HLE_weighted_sum
        f2 = abs(SII(MCS, EPR))
        f3 = x[20] + x[21]
        
        out["F"] = [f1, f2, f3]
        # out["G"] = [g2, g3, g4, g5, g6, g7]
        out["G"] = [g1, g2, g3, g4, g5, g6, g7]
        
        
problem = MyProblem()

algorithm = NSGA2(pop_size=100)
res = minimize(problem,
               algorithm,
               ('n_gen', 500),
               seed=17,
               verbose=True,
               save_history = True)

t_end = time.time()

X = res.X
F = res.F
G = res.G
   
n_evals = np.array([e.evaluator.n_eval for e in res.history])
opt_G = np.array([e.opt[0].G for e in res.history])
opt_X = np.array([e.opt[0].X for e in res.history])

plt.plot(range(0, len(opt_G)), opt_G)
plt.xlabel("Generations")
plt.ylabel("G constraint")
plt.show()

print('Time taken: ' + str(t_end - t0) + ' seconds')

u = np.zeros((X.shape[0], 20))
u_transformed = np.zeros((F.shape[0], 20))
for m in range(0, X.shape[0]):
    u[m, :] = x_to_u(X[m, :], beta, gamma, mu, sigma)
for m in range(0, F.shape[0]):
    for n in range(0, 10):
        u_transformed[m, n*2] = inv_transforms_z(u[m, n*2], mu[0], sigma[0], 1)
        u_transformed[m, n*2 + 1] = inv_transforms_z(u[m, n*2 + 1], mu[2], sigma[2], 1)


# Plotting...
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(-F[:, 0], F[:, 1], F[:, 2])
plt.title("Objective space")
ax.set_xlabel("f_1: maximise weighted average HLE")
ax.set_ylabel("f_2: minimise HLE inequality")
ax.set_zlabel("f_3: minimise cost")
plt.show()


fig, ([ax1, ax2]) = plt.subplots(2, 1, figsize =(9, 6))

ax1.boxplot(u_transformed[:, 1:20:2])
ax1.set_xticklabels(xticklabels)
ax1.set_ylabel('Adult skills')

ax2.boxplot(u_transformed[:, 0:20:2])
ax2.set_xticklabels(xticklabels)
ax2.set_ylabel('Labour remuneration')
plt.show()

cluster_A = [Cl4_A, Cl3_A, Cl4_A, Cl4_A, Cl4_A, Cl4_A, Cl3_A, Cl4_A, Cl2_A, Cl3_A]
cluster_B = [Cl4_B, Cl3_B, Cl4_B, Cl4_B, Cl4_B, Cl4_B, Cl3_B, Cl4_B, Cl2_B, Cl3_B]
cluster_int = [Cl4_intercept, Cl3_intercept, Cl4_intercept, Cl4_intercept, Cl4_intercept, Cl4_intercept, Cl3_intercept, Cl4_intercept, Cl2_intercept, Cl3_intercept]

#%%
# =============================================================================
# if save_results == True:
# =============================================================================
LA_out = [None] * 10
uUpdated = np.zeros((100, 3))
EPR = np.zeros((len(X), 10))
PVT = np.zeros((len(X), 10))
FPR = np.zeros((len(X), 10))
DHI = np.zeros((len(X), 10))
EIQ = np.zeros((len(X), 10))
MCS = np.zeros((len(X), 10))
PMR = np.zeros((len(X), 10))
HLE = np.zeros((len(X), 10))
cluster_A = [Cl4_A, Cl3_A, Cl4_A, Cl4_A, Cl4_A, Cl4_A, Cl3_A, Cl4_A, Cl2_A, Cl3_A]
cluster_B = [Cl4_B, Cl3_B, Cl4_B, Cl4_B, Cl4_B, Cl4_B, Cl3_B, Cl4_B, Cl2_B, Cl3_B]
cluster_int = [Cl4_intercept, Cl3_intercept, Cl4_intercept, Cl4_intercept, Cl4_intercept, Cl4_intercept, Cl3_intercept, Cl4_intercept, Cl2_intercept, Cl3_intercept]
uAll = np.zeros(3)
for m in range(0, len(X)):
    for n in range(0, 10):
        u2 = x_to_u(X[m, :], beta, gamma, mu, sigma)
# =============================================================================
#         LA1_yout1, LA1_yout2 = solver(u2[n*2:n*2 + 2], cluster_A[n], cluster_B[n], cluster_int[n], baseline_JPC[n])
#         EPR[m, n] = inv_transforms_z(LA1_yout1, mu[3], sigma[3], 1)   # z1
#         PVT[m, n] = inv_transforms_z(LA1_yout2[4], mu[4], sigma[4], 2) # z2
#         FPR[m, n] = inv_transforms_z(LA1_yout2[5], mu[5], sigma[5], 3) # z3
#         DHI[m, n] = inv_transforms_z(LA1_yout2[6], mu[6], sigma[6], 4) # z4
#         EIQ[m, n] = inv_transforms_z(LA1_yout2[7], mu[7], sigma[7], 5) # z5
#         MCS[m, n] = inv_transforms_z(LA1_yout2[2], mu[8], sigma[8], 6) # z6
#         PMR[m, n] = inv_transforms_z(LA1_yout2[8], mu[9], sigma[9], 7) # z7
#         HLE[m, n] = inv_transforms_z(LA1_yout2[3], mu[10], sigma[10], 8) # z8
# =============================================================================
        uAll[0] = u[m, 2*n]
        uAll[1] = nonlinear_transforms_z(baseline_JPC[n], mu[1], sigma[1], 2)
        uAll[2] = u[m, 2*n + 1]
        LA_out[n] = full_model(uAll, cluster_A[n], cluster_B[n], cluster_int[n])
        EPR[m, n] = inv_transforms_z(LA_out[n][0], mu[3], sigma[3], 1) # z1
        PVT[m, n] = inv_transforms_z(LA_out[n][1], mu[4], sigma[4], 2) # z2
        FPR[m, n] = inv_transforms_z(LA_out[n][2], mu[5], sigma[5], 3) # z2
        DHI[m, n] = inv_transforms_z(LA_out[n][3], mu[6], sigma[6], 4) # z2
        EIQ[m, n] = inv_transforms_z(LA_out[n][4], mu[7], sigma[7], 5) # z2
        MCS[m, n] = inv_transforms_z(LA_out[n][5], mu[8], sigma[8], 6) # z2
        PMR[m, n] = inv_transforms_z(LA_out[n][6], mu[9], sigma[9], 7) # z2
        HLE[m, n] = inv_transforms_z(LA_out[n][7], mu[10], sigma[10], 8) # z2
    # plot_SII(MCS[m, :], PVT[m, :], pops)

print(MCS[0,0])
#%%

a = res.history
G_historical = np.zeros((100, 100))
x1 = np.zeros((100, 500))
for k in range(0, 22):
    for n in range(0, 500):
        for m in range(0, 100):
            x1[m,n] = a[n].pop[m].X[k]
    pd.DataFrame(x1).to_csv('Results/x' + str(k + 1) + '.csv', header = False, index = False)
        
N = F.shape[0] - 1
a = res.history
x1 = np.zeros((F.shape[0], 22))
f1 = np.zeros((F.shape[0], 3))
for m in range(0, F.shape[0]):
    f1[m, :] = a[N].pop[m].F
    for n in range(0, 22):
    
        x1[m,n] = a[N].pop[m].X[n]
    
    

if save_results == True:
    pd.DataFrame(EPR).to_csv('Results/EPR.csv', header = False, index = False)
    pd.DataFrame(PVT).to_csv('Results/PVT.csv', header = False, index = False)
    pd.DataFrame(FPR).to_csv('Results/FPR.csv', header = False, index = False)
    pd.DataFrame(DHI).to_csv('Results/DHI.csv', header = False, index = False)
    pd.DataFrame(EIQ).to_csv('Results/EIQ.csv', header = False, index = False)
    pd.DataFrame(MCS).to_csv('Results/MCS.csv', header = False, index = False)
    pd.DataFrame(PMR).to_csv('Results/PMR.csv', header = False, index = False)
    pd.DataFrame(HLE).to_csv('Results/HLE.csv', header = False, index = False)
    pd.DataFrame(u_transformed[:, 0:20:2]).to_csv('Results/PLW.csv', header = False, index = False)
    pd.DataFrame(u_transformed[:, 1:20:2]).to_csv('Results/SAQ.csv', header = False, index = False)
    pd.DataFrame(opt_G).to_csv('Results/G_constraint.csv', header = False, index = False)
    
    pd.DataFrame(F).to_csv('Results/Feval.csv', header = False, index = False)
    pd.DataFrame(X).to_csv('Results/X.csv', header = False, index = False)
    
    pd.DataFrame(x1).to_csv('Results/X_final.csv', header = False, index = False)
    pd.DataFrame(f1).to_csv('Results/F_final.csv', header = False, index = False)

