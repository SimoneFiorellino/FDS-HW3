import numpy as np
import pandas as pd
from tqdm import tqdm


class SVM:
    
    def __init__(self, x, y, C=1, tol=0.001, eps=0.001, kernel='linear', sigma=0.45):
        """
        :param x -> train data
        :param y -> train labels
        :param C -> soft margin
        :param tol -> tolerance 
        :param eps -> epsilon
        :param kernel -> either 'linear' or 'gaussian'
        :param sigma -> variance, for gaussian kernel only
        """

        self.x = x
        self.y = y
        self.alpha = None
        self.support_vectors_ = None # vector of support vectors indices
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


    def _check_bounds(self):
        """
        Check if the KKT conditions are met. 
        
        Return the indices of the points inside the margin 
        """

        return np.flatnonzero(np.logical_and(self.alpha > 0, self.alpha < self.C))


    def _compute_L_H(self, i, j):
        """
        Compute L and H according to the formulation in Platt's paper
        """

        if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[j])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
            return L, H
        L = max(0, self.alpha[j] + self.alpha[i] - self.C)
        H = min(self.C, self.alpha[j] + self.alpha[i])
        return L, H


    def _examine_example(self, j):
        """
        Search for alphas to be optimized

        Return either changed (1) or unchanged (0)
        """

        self.E[j] = self._error(j)
        r_j = self.E[j] * self.y[j]
        if (r_j < -self.tol and self.alpha[j] < self.C) or (r_j > self.tol and self.alpha[j] > 0):
            non_0_C = self._check_bounds()
            non_0_C_len =  len(non_0_C)
            if non_0_C_len > 1:
                if self._second_heuristicA(j, non_0_C):
                    return 1
                if self._second_heuristicB(j, non_0_C_len, non_0_C):
                    return 1
            if self._second_heuristicB(j, self.m):
                return 1
        return 0


    def _error(self, i):
        """
        Compute the error of the prediction as (prediction - ground-truth label)

        Return the error (scalar)
        """

        return self.decision_function(self.x[i]) - self.y[i]


    def _first_heuristic(self, num_changed):
        """
        First heuristic according to Platt's paper

        Return the number of alphas changed
        """

        ex_not_bounds = self._check_bounds() # array of indexes
        for i in ex_not_bounds:
            num_changed += self._examine_example(i)
        return num_changed


    def _get_support_vectors(self):
        """
        Get the indexes of the support vectors

        Return numpy.array of indexes
        """

        if self.support_vectors_ is None:
            return np.flatnonzero(self.alpha != 0)
        return self.support_vectors_


    def _initialize_parameters(self):
        """
        Initialize parameters before optimization
        """

        self.alpha = np.zeros((self.m, 1))
        self.b = 0
        self.E = self.y * -1 # result of the decision function with all alphas equal to zero


    def _main_smo_fun(self):
        """
        Main routine of the SMO algorithm
        """

        self._initialize_parameters()
        num_changed = 0 
        examine_all = True
        with tqdm() as pbar:
            while num_changed > 0 or examine_all:
                num_changed = 0
                if examine_all:
                    for i in range(self.m):
                        num_changed += self._examine_example(i)
                else:
                    num_changed = self._first_heuristic(num_changed)

                if examine_all == True:
                    examine_all = False
                elif num_changed == 0:
                    examine_all = True
                pbar.update(1)


    def _objective_function(self):
        """
        Objective function of the dual representation

        Return the value of the objective function (scalar) 
        """

        support_vectors_idxs = self._get_support_vectors()
        result = 0
        for i in support_vectors_idxs:
            for j in support_vectors_idxs:
                result += self.y[i] * self.y[j] * self.alpha[i] * self.alpha[j] * self.kernel(self.x[i], self.x[j])
        result = 0.5 * result - np.sum(self.alpha[support_vectors_idxs])
        return result


    def _second_heuristicA(self, j, non_bounds):
        """
        Search for the candidate alpha that maximizes the in-step progress
        """

        if self.E[j] > 0:
            i = np.argmin(self.E[non_bounds])
        else:
            i = np.argmax(self.E[non_bounds])
        if self._take_step(i, j):
            return True
        return False


    def _second_heuristicB(self, j, s_len, arr=None):
        """
        Search systematically the alphas 
        """

        if arr is None:
            arr = np.arange(0, s_len)
        for i in np.roll(arr, np.random.randint(0, s_len)):
            if self._take_step(i, j):
                return True
        return False


    def _take_step(self, i, j):
        """
        Compute the optimzation step and update the parameters

        Return True or False
        """

        if i == j:
            return False

        self.E[i] = self.predict(self.x[i, :]) - self.y[i]
        s = self.y[i] * self.y[j]
        L, H = self._compute_L_H(i, j)
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
            Lobj = self._objective_function()
            self.alpha[j] = H
            Hobj = self._objective_function()
            self.alpha[j] = aj_old
            if Lobj < Hobj - self.eps:
                aj_new = L
            elif Lobj > Hobj + self.eps:
                aj_new = H
            else: 
                aj_new = self.alpha[j]
        if abs(aj_new - self.alpha[j]) < (self.eps * (aj_new + self.alpha[j] + self.eps)):
            return False
        ai_new = self.alpha[i] + s * (self.alpha[j] - aj_new)
        if 0 < ai_new < self.C:
            self.b = self.b + self.E[i] + self.y[i] * (ai_new - self.alpha[i]) * self.kernel(self.x[i], self.x[i]) + self.y[j] * (aj_new - self.alpha[j]) * self.kernel(self.x[i], self.x[j])
        elif 0 < aj_new < self.C:
            self.b = self.b + self.E[j] + self.y[i] * (ai_new - self.alpha[i]) * self.kernel(self.x[i], self.x[j]) + self.y[j] * (aj_new - self.alpha[j]) * self.kernel(self.x[j], self.x[j])
        else:
            b1 = self.b + self.E[i] + self.y[i] * (ai_new - self.alpha[i]) * self.kernel(self.x[i], self.x[i]) + self.y[j] * (aj_new - self.alpha[j]) * self.kernel(self.x[i], self.x[j])
            b2 = self.b + self.E[j] + self.y[i] * (ai_new - self.alpha[i]) * self.kernel(self.x[i], self.x[j]) + self.y[j] * (aj_new - self.alpha[j]) * self.kernel(self.x[j], self.x[j])
            self.b = (b1 + b2) / 2
        self.alpha[i] = ai_new
        self.alpha[j] = aj_new
        self.E[i] = self._error(i)
        self.E[j] = self._error(j)
        return True


    def decision_function(self, x):
        """
        Compute the decision function sum(y @ alpha * kernel) - b

        Return scalar
        """

        u = 0
        support_vectors_idxs = self._get_support_vectors()
        for i in support_vectors_idxs:
            u += self.y[i] * self.alpha[i] * self.kernel(self.x[i], x)
        u = u - self.b
        return u


    def fit(self):
        """
        Fit the data and start parameters optimization
        """

        self._main_smo_fun()
        self.support_vectors_ = self._get_support_vectors()

    
    def predict(self, x):
        """
        Predict the class given the input x

        Return either 1 or -1
        """ 

        return int(np.sign(self.decision_function(x)))


    def support_vectors(self):
        """
        Return the alphas of the support vectors
        """

        return self.alpha[self.support_vectors_]




import time

train_set = pd.read_csv('./data/train.csv')
test_set = pd.read_csv('./data/test.csv')

train_set.drop('subject', axis=1, inplace=True)
test_set.drop('subject', axis=1, inplace=True)

labels_train = train_set.Activity
labels_test = test_set.Activity

train_set.drop('Activity', axis=1, inplace=True)
test_set.drop('Activity', axis=1, inplace=True)

numeric_labels = {'STANDING':-1, 'SITTING':1, 'LAYING':1, 'WALKING':1, 'WALKING_DOWNSTAIRS':1, 'WALKING_UPSTAIRS':1}
labels_train = labels_train.map(lambda x: numeric_labels[x])
labels_test = labels_test.map(lambda x: numeric_labels[x])


x_train = train_set.to_numpy()
x_test = test_set.to_numpy()

y_train = labels_train.to_numpy()
y_test = labels_test.to_numpy()

svm = SVM(x_train[:, :10], y_train, kernel='gaussian', C=1, tol=0.001, eps=0.001)

t0 = time.time()
svm.fit()

predictions_list = []
for i in range(y_test.shape[0]):
    predictions_list.append(svm.predict(x_test[i, :10]))
predictions = np.array(predictions_list)

t = time.time()

print(np.sum(np.equal(predictions, y_test)))
print(f'\nTime taken: {t - t0}')

# print(f'Predicted: {svm.predict(x[95])}\tGround truth: {y[95]}\n')
# print(f'Decision function: {svm.decision_function(x[95])}')
# print(f'Errors: {svm.E}')


# df_x = pd.read_csv("./data/logistic_x.txt", sep=" +", names=["x1","x2"], header=None, engine='python')
# df_y = pd.read_csv('./data/logistic_y.txt', sep=' +', names=["y"], header=None, engine='python')
# df_y = df_y.astype(int)

# x = np.hstack([np.ones((df_x.shape[0], 1)), df_x[["x1","x2"]].values])
# y = df_y["y"].values

# # np.random.seed(17349)

# svm = SVM(x, y, kernel='gaussian', C=0.5, tol=0.001, eps=0.001)

# svm.fit()

# predictions_list = []
# for i in range(99):
#     predictions_list.append(svm.predict(x[i]))
# predictions = np.array(predictions_list)
# print(predictions)
# print(np.sum(np.equal(predictions, y)))