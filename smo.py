 

# def take_step(i1, i2, alpha, y, placeholder_svm, C, X):
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
#     k11 = kernel(X[i1], X[i1])
#     k12 = kernel(X[i1], X[i2])
#     k22 = kernel(X[i2], X[i2])
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
#         b = b - E1 - y1 * (a1 - alpha[i1]) * kernel(X[i1], X[i1]) - y[i2] * (a2 - alpha[i2]) * kernel(X[i1], X[i2])
#     elif 0 < a2 < C:
#         b = b - E2 - y1 * (a1 - alpha[i1]) * kernel(X[i1], X[i2]) - y[i2] * (a2 - alpha[i2]) * kernel(X[i2], X[i2])
#     else:
#         if L != H:
#            b1 = b - E1 - y1 * (a1 - alpha[i1]) * kernel(X[i1], X[i1]) - y[i2] * (a2 - alpha[i2]) * kernel(X[i1], X[i2]) 
#            b2 = b - E2 - y1 * (a1 - alpha[i1]) * kernel(X[i1], X[i2]) - y[i2] * (a2 - alpha[i2]) * kernel(X[i2], X[i2])
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


def kernel(x_i, x_j):
    return np.sum(np.dot(x_i, x_j))


def gamma(u, X, y, alpha, b):
    result = 0
    m = X.shape[0]
    for i in range(0, m):
        result += alpha[i] * y[i] * kernel(X[i], u)
    return result + b

def error(x_i, y_i, x, y, alpha, b):
    return gamma(x_i, x, y, alpha, b) - y_i


def smo(C, tol, max_passes, X, y, eps):
    m = X.shape[0]
    E = np.zeros((m, 1))
    alpha = np.zeros((m, 1))
    b = 0 
    passes = 0
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            E[i] = gamma(X[i, :], X, y, alpha, b) - y[i]
            if ((y[i] * E[i] < -tol) and alpha[i] < C) or ((y[i] * E[i] > tol) and alpha[i] > 0):
                j = np.random.randint(0, m)
                while j == i:
                    j = np.random.randint(0, m)
                E[j] = gamma(X[j, :], X, y, alpha, b) - y[j]
                ai_old, aj_old = alpha[i], alpha[j]
                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                if L == H:
                    continue
                eta = 2 * kernel(X[i], X[j]) - kernel(X[i], X[i]) - kernel(X[j], X[j])
                if eta >= 0:
                    continue
                aj_new = aj_old - y[j] * (E[i] - E[j]) / eta
                if aj_new > H:
                    aj_new = H
                elif aj_new < L:
                    aj_new = L
                if abs(aj_new - aj_old) < eps:
                    continue
                ai_new = ai_old + y[i] * y[j] * (aj_old - aj_new)
                if 0 < ai_new < C:
                    b = b - E[i] - y[i] * (ai_new - ai_old) * kernel(X[i], X[i]) - y[j] * (aj_new - aj_old) * kernel(X[i], X[j])
                elif 0 < aj_new < C:
                    b = b - E[j] - y[i] * (ai_new - ai_old) * kernel(X[i], X[j]) - y[j] * (aj_new - aj_old) * kernel(X[j], X[j])
                else:
                    b1 = b - E[i] - y[i] * (ai_new - ai_old) * kernel(X[i], X[i]) - y[j] * (aj_new - aj_old) * kernel(X[i], X[j])
                    b2 = b - E[j] - y[i] * (ai_new - ai_old) * kernel(X[i], X[j]) - y[j] * (aj_new - aj_old) * kernel(X[j], X[j])
                    b = (b1 + b2) / 2
                alpha[i] = ai_new
                alpha[j] = aj_new
                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    return alpha, b


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

alpha, b = smo(1, 0.00001, 300, x, y, 0.00001)
print(alpha)
print()
print(b)

print(x[98, :])
print(y[98])

print(gamma(x[98], x, y, alpha, b))



from sklearn.svm import SVC


svc = SVC(kernel='linear', tol=0.00001, max_iter=300)
svc.fit(x, y)
test = x[98]
test = test.reshape(1, -1)
print(svc.predict(test))
print(svc.decision_function(test))