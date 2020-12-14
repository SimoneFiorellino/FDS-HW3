 

# def take_step(i1, i2, alpha, y, placeholder_svm, C, x):
#     if i1 == i2:
#         return 0

#     alph1 = alpha[i1]
#     y1 = y[i1]
#     E1 = placeholder_svm(i1) - y1
#     s = y1 * y[i2]
#     if y1 != y[i2]:
#         L = max(0, alph1 - alpha[i2])
#         H = min(C, C + alpha[i2] - alph1)
#     else:
#         L = max(0, alpha[i2] + alph1 - C)
#         H = min(C, alpha[i2] - alph1)
#     if L == H:
#         return 0
#     k11 = self.kernel(x[i1], x[i1])
#     k12 = self.kernel(x[i1], x[i2])
#     k22 = self.kernel(x[i2], x[i2])
#     eta = k11 + k22 - 2 * k12
#     if eta > 0: 
#         a2 = alpha[i2] + y[i2] * (E1 - E2) / eta
#         if a2 < L:
#             a2 = L
#         elif a2 > H:
#             a2 = H
#     else:
#         Lobj = objective_f(L)
#         Hobj = objective_f(H)
#         if Lobj < Hobj - eps:
#             a2 = L
#         elif Lobj > Hobj + eps:
#             a2 = H
#         else:
#             a2 = alpha[i2]
#     if abs(a2 - alpha[i2] < eps * (a2 + alpha[i2] + eps)):
#         return 0
#     a1 = alpha[i1] + s * (alpha[i2] - a2)
#     if 0 < a1 < C:
#         b = b - E1 - y1 * (a1 - alpha[i1]) * self.kernel(x[i1], x[i1]) - y[i2] * (a2 - alpha[i2]) * self.kernel(x[i1], x[i2])
#     elif 0 < a2 < C:
#         b = b - E2 - y1 * (a1 - alpha[i1]) * self.kernel(x[i1], x[i2]) - y[i2] * (a2 - alpha[i2]) * self.kernel(x[i2], x[i2])
#     else:
#         if L != H:
#            b1 = b - E1 - y1 * (a1 - alpha[i1]) * self.kernel(x[i1], x[i1]) - y[i2] * (a2 - alpha[i2]) * self.kernel(x[i1], x[i2]) 
#            b2 = b - E2 - y1 * (a1 - alpha[i1]) * self.kernel(x[i1], x[i2]) - y[i2] * (a2 - alpha[i2]) * self.kernel(x[i2], x[i2])
#            b = (b1 + b2) / 2
#     E1 = placeholder_svm(i1) - y1
#     E2 = placeholder_svm(i2) - y[i2]
#     alpha[i1] = a1
#     alpha[i2] = a2
#     return 1


# def examine_example(i2):
#     y2 = y[i2]
#     alph2 = alpha[i2]
#     E2 = placeholder_svm(i2) - y2
#     r2 = E2 * y2
#     if (r2 < -tol and alph2 < C) or (r2 > tol and alph2 > 0):
#         if n_non_zero_C > 1: 
#             i1 = 


import numpy as np


class SVM:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.alpha = None
        self.b = None


    def kernel(self, x_i, x_j):
        return np.sum(np.dot(x_i, x_j))


    def gamma(self, u):
        result = 0
        m = self.x.shape[0]
        for i in range(0, m):
            result += self.alpha[i] * self.y[i] * self.kernel(self.x[i], u)
        return result + self.b


    def _error(self, x_i, y_i):
        return self.gamma(x_i) - y_i


    def simplified_smo(self, C, tol, eps, max_passes):
        m = self.x.shape[0]
        E = np.zeros((m, 1))
        self.alpha = np.zeros((m, 1))
        self.b = 0 
        passes = 0
        while passes < max_passes:
            num_changed_alphas = 0
            for i in range(m):
                E[i] = self.gamma(self.x[i, :]) - self.y[i]
                if ((self.y[i] * E[i] < -tol) and self.alpha[i] < C) or ((self.y[i] * E[i] > tol) and self.alpha[i] > 0):
                    j = np.random.randint(0, m)
                    while j == i:
                        j = np.random.randint(0, m)
                    E[j] = self.gamma(x[j, :]) - self.y[j]
                    ai_old, aj_old = self.alpha[i], self.alpha[j]
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(C, C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - C)
                        H = min(C, self.alpha[i] + self.alpha[j])
                    if L == H:
                        continue
                    eta = 2 * self.kernel(self.x[i], self.x[j]) - self.kernel(self.x[i], self.x[i]) - self.kernel(self.x[j], self.x[j])
                    if eta >= 0:
                        continue
                    aj_new = aj_old - self.y[j] * (E[i] - E[j]) / eta
                    if aj_new > H:
                        aj_new = H
                    elif aj_new < L:
                        aj_new = L
                    if abs(aj_new - aj_old) < eps:
                        continue
                    ai_new = ai_old + self.y[i] * self.y[j] * (aj_old - aj_new)
                    if 0 < ai_new < C:
                        self.b = self.b - E[i] - self.y[i] * (ai_new - ai_old) * self.kernel(self.x[i], self.x[i]) - self.y[j] * (aj_new - aj_old) * self.kernel(self.x[i], self.x[j])
                    elif 0 < aj_new < C:
                        self.b = self.b - E[j] - self.y[i] * (ai_new - ai_old) * self.kernel(self.x[i], self.x[j]) - self.y[j] * (aj_new - aj_old) * self.kernel(self.x[j], self.x[j])
                    else:
                        b1 = self.b - E[i] - self.y[i] * (ai_new - ai_old) * self.kernel(self.x[i], self.x[i]) - self.y[j] * (aj_new - aj_old) * self.kernel(self.x[i], self.x[j])
                        b2 = self.b - E[j] - self.y[i] * (ai_new - ai_old) * self.kernel(self.x[i], self.x[j]) - self.y[j] * (aj_new - aj_old) * self.kernel(self.x[j], self.x[j])
                        self.b = (b1 + b2) / 2
                    self.alpha[i] = ai_new
                    self.alpha[j] = aj_new
                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        return self.alpha, self.b
        

    def _examine_example(self, j, C, tol):
        y_j = self.y[j]
        alpha_j = self.alpha[j]
        self.E[j] = self.gamma(self.x[j, :]) - y_j
        r_j = self.E[j] * y_j
        if (r_j < -tol and alpha_j < C) or (r_j > tol and alpha_j > 0):
            ex_not_bounds = self._check_bounds(C)
            if len(ex_not_bounds) > 1:
                i = self._check_bounds(j)


    def _choose_i(self, j):
        


    
    def _check_bounds(self, C): # first heuristic
        return np.flatnonzero(np.logical_and(self.alpha > 0, self.alpha < C))


    def complete_smo(self, C, tol, eps, max_passes):
        self.E = self.y * -1
        m = self.x.shape[0]
        num_changed = 0 
        examine_all = True
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in m:
                    num_changed += self._examine_example(i, C, tol)
            else: # first heuristic
                ex_not_bounds = self._check_bounds(C) # array of indexes
                for i in ex_not_bounds:
                    num_changed += self._examine_example(i, C, tol)
            if examine_all == True:
                examine_all = False
            elif num_changed == 0:
                examine_all = True


import numpy as np # imports a fast numerical programming library
import scipy as sp # imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm # allows us easy access to colormaps
import matplotlib.pyplot as plt # sets up plotting under plt
import pandas as pd # lets us handle data as dataframes

# sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

import seaborn as sns # sets up styles and gives us more plotting options


df_x = pd.read_csv("./data/logistic_x.txt", sep="\ +", names=["x1","x2"], header=None, engine='python')
df_y = pd.read_csv('./data/logistic_y.txt', sep='\ +', names=["y"], header=None, engine='python')
df_y = df_y.astype(int)
print(df_x.head())

x = np.hstack([np.ones((df_x.shape[0], 1)), df_x[["x1","x2"]].values])
y = df_y["y"].values


svm = SVM(x, y)

alpha, b = svm.simplified_smo(C=1, tol=0.00001, eps=0.00001, max_passes=50)
print(alpha)
print()
print(b)

print(x[98, :])
print(y[98])

print(svm.gamma(x[98]))



from sklearn.svm import SVC


svc = SVC(kernel='linear', tol=0.00001, max_iter=300)
svc.fit(x, y)
test = x[98]
test = test.reshape(1, -1)
print(svc.predict(test))
print(svc.decision_function(test))