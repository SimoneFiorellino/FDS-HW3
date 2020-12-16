import numpy as np
import pandas as pd # lets us handle data as dataframes




class SVM:
    
    def __init__(self, x, y, C=1, tol=0.001, eps=0.001, kernel='linear', sigma=0.45):
        self.x = x
        self.y = y
        self.alpha = None
        self.b = 0
        self.m = x.shape[0]
        self.C = C
        self.tol = tol
        self.eps = eps
        self.simga = sigma
        kernel_type = {
            'linear': lambda x_i, x_j: np.sum(np.dot(x_i, x_j)),
            'gaussian': lambda x_i, x_j: np.exp(- np.square(np.linalg.norm(x_i - x_j)) / 2*(sigma**2))
        }
        self.kernel = kernel_type[kernel]



    # def kernel(self, x_i, x_j):
    #     return np.sum(np.dot(x_i, x_j))


    def gamma(self, u):
        result = 0
        m = self.x.shape[0]
        for i in range(0, m):
            result += self.alpha[i] * self.y[i] * self.kernel(self.x[i], u)
        return result + self.b


    def objective_function(self):
        alpha_non_zero = np.flatnonzero(self.alpha != 0)
        result = 0
        for i in alpha_non_zero:
            for j in alpha_non_zero:
                result += self.y[i] * self.y[j] * self.alpha[i] * self.alpha[j] * self.kernel(self.x[i], self.x[j])
        result -= np.sum(self.alpha[alpha_non_zero])
        return result


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


    def _compute_L_H(self, i, j, C):
        if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[j])
            H = min(C, C + self.alpha[j] - self.alpha[i])
            return L, H
        L = max(0, self.alpha[j] + self.alpha[i] - C)
        H = min(C, self.alpha[j] + self.alpha[i])
        return L, H


    def take_step(self, i, j, C, eps):
        if i == j:
            return False
        self.E[i] = self.gamma(self.x[i, :]) - self.y[i]
        s = self.y[i] * self.y[j]
        L, H = self._compute_L_H(i, j, C)
        if L == H:
            return False
        eta = self.kernel(self.x[i], self.x[i]) + self.kernel(self.x[j], self.x[j]) - 2 * self.kernel(self.x[i], self.x[j]) 
        if eta > 0:
            aj_new = self.alpha[j] + self.y[j] * (self.E[i] - self.E[j]) / eta
            if aj_new < L:
                aj_new = L
            elif aj_new > H:
                aj_new = H 
        else:
            aj_old = self.alpha[j]
            self.alpha[j] = L
            Lobj = self.objective_function()
            self.alpha[j] = H
            Hobj = self.objective_function()
            self.alpha[j] = aj_old
            if Lobj < Hobj - eps:
                aj_new = L
            elif Lobj > Hobj + eps:
                aj_new = H
            else: 
                aj_new = self.alpha[j]
        if abs(aj_new - self.alpha[j]) < eps * (aj_new + self.alpha[j] + eps):
            return False
        ai_new = self.alpha[i] + s * (self.alpha[j] - aj_new)
        if 0 < ai_new < C:
            self.b = self.b - self.E[i] - self.y[i] * (ai_new - self.alpha[i]) * self.kernel(self.x[i], self.x[i]) - self.y[j] * (aj_new - self.alpha[j]) * self.kernel(self.x[i], self.x[j])
        elif 0 < aj_new < C:
            self.b = self.b - self.E[j] + self.y[i] * (ai_new - self.alpha[i]) * self.kernel(self.x[i], self.x[j]) - self.y[j] * (aj_new - self.alpha[j]) * self.kernel(self.x[j], self.x[j])
        else:
            b1 = self.b - self.E[i] - self.y[i] * (ai_new - self.alpha[i]) * self.kernel(self.x[i], self.x[i]) - self.y[j] * (aj_new - self.alpha[j]) * self.kernel(self.x[i], self.x[j])
            b2 = self.b - self.E[j] - self.y[i] * (ai_new - self.alpha[i]) * self.kernel(self.x[i], self.x[j]) - self.y[j] * (aj_new - self.alpha[j]) * self.kernel(self.x[j], self.x[j])
            self.b = (b1 + b2) / 2
        self.E[i] = self._error(self.x[i], self.y[i])
        self.E[j] = self._error(self.x[j], self.y[j])
        self.alpha[i] = ai_new
        self.alpha[j] = aj_new
        return True


    def _examine_example(self, j, C, tol, eps):
        y_j = self.y[j]
        alpha_j = self.alpha[j]
        self.E[j] = self.gamma(self.x[j, :]) - y_j
        r_j = self.E[j] * y_j
        if (r_j < -tol and alpha_j < C) or (r_j > tol and alpha_j > 0):
            ex_not_bounds = self._check_bounds(C)
            ex_not_bounds_len =  len(ex_not_bounds)
            if ex_not_bounds_len > 1:
                i = self._choose_i(j, ex_not_bounds)
                if self.take_step(i, j, C, eps):
                    return 1
                random_point = np.random.randint(0, ex_not_bounds_len)
                for k in range(0, ex_not_bounds_len):
                    i = (k + random_point) % ex_not_bounds_len
                    if self.take_step(i, j, C, eps):
                        return 1
            random_point = np.random.randint(0, self.m)
            for k in range(0, self.m):
                i = (k + random_point) % self.m
                if self.take_step(i, j, C, eps):
                    return 1
        return 0


    def _choose_i(self, j, non_bounds):
        if self.E[j] > 0:
            return np.argmin(self.E[non_bounds])
        return np.argmax(self.E[non_bounds])


    def _check_bounds(self, C): # first heuristic
        return np.flatnonzero(np.logical_and(self.alpha > 0, self.alpha < C))


    def complete_smo(self, C, tol, eps):
        m = self.x.shape[0]
        self.E = self.y * -1
        self.alpha = np.zeros((m, 1))
        self.b = 0
        num_changed = 0 
        examine_all = True
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(m):
                    num_changed += self._examine_example(i, C, tol, eps)
            else: # first heuristic
                ex_not_bounds = self._check_bounds(C) # array of indexes
                for i in ex_not_bounds:
                    num_changed += self._examine_example(i, C, tol, eps)
            if examine_all == True:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
        return self.alpha, self.b



df_x = pd.read_csv("./data/logistic_x.txt", sep=" +", names=["x1","x2"], header=None, engine='python')
df_y = pd.read_csv('./data/logistic_y.txt', sep=' +', names=["y"], header=None, engine='python')
df_y = df_y.astype(int)
print(df_x.head())

x = np.hstack([np.ones((df_x.shape[0], 1)), df_x[["x1","x2"]].values])
y = df_y["y"].values

# np.random.seed(17349)

svm = SVM(x, y, kernel='linear')

alpha, b = svm.complete_smo(C=1, tol=0.001, eps=0.001)
print()
print(alpha)
print()
print(b)
print()
print(svm.gamma(x[5]))
# alpha, b = svm.simplified_smo(C=1, tol=0.00001, eps=0.00001, max_passes=50)
# print(alpha)
# print()
# print(b)

# print(x[98, :])
# print(y[98])

# print(svm.gamma(x[98]))



from sklearn.svm import SVC


svc = SVC(kernel='linear', tol=0.001, max_iter=300)
svc.fit(x, y)
test = x[5]
test = test.reshape(1, -1)
print(svc.predict(test))
print(svc.decision_function(test))