# This script implements soft margin Support vector machine classifier for two classes
# using GAMS as the optimization solver
# Wei Jiang
# 11/26/2016

import numpy as np
from gams import *
import os
import sys
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
class svm():
    """
    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Indices of support vectors.
    support_vectors_ : array-like, shape = [n_SV, n_features]
        Support vectors.
    n_support_ : array-like, dtype=int32, shape = [n_class]
        Number of support vectors for each class.
    dual_coef_ : array, shape = [n_class-1, n_SV]
        Coefficients of the support vector in the decision function.
        alpha, dual variable for inequality margin constraint
    coef_ : array, shape = [n_class-1, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.
        `coef_` is a readonly property derived from `dual_coef_` and
        `support_vectors_`.
    intercept_ : array, shape = [n_class * (n_class-1) / 2]
        Constants in decision function.
    """

    def __init__(self, C = 2 ):
        self.C = C
        self.support = []
        self.support_vectors_ = None
        self.non_margin_support_vectors = None
        self.n_support_ = []
        self.dual_coef_ = []
        self.coef_ = []
        self.intercept = None

    def _get_model_text(self):
        return '''
Sets
i number of data samples
;
Alias(i,z);
Parameters
y(i) target label
K_y(i,z) y kernel
K_x(i,z) kernel or inner product
;
Scalar
C regularization parameter
;
$if not set gdxincname $abort 'no include file name for data file provided'
$GDXin %gdxincname%
$load i y K_y K_x C
$GDXin
;
Positive Variable
alpha(i)
;
Variable
L  lagrangian of the dual problem
;
Equations
obj     objective function
dual_cap(i)
cons_equal;

obj ..  L =e= sum(i, alpha(i)) - 0.5 * sum(i, sum(z, alpha(i)*alpha(z)*K_y(i,z)*K_x(i,z) ));
dual_cap(i) ..  alpha(i) =l= C;
cons_equal ..       sum(i, alpha(i) * y(i)) =e= 0;

Model svm_dual /all/;
Solve svm_dual using QCP maximizing L;
'''

    def fit(self, X, y, solve_dual = True):
        """
        X: numpy array, [number of sample, NO of features], feature already scaled
        y: array of -1 or 1
        solve_dual: boolean, if true, solve the quadratic dual problem
                    otherwise, solve the primal problem
        """
        self.number_sample = X.shape[0]
        ws = GamsWorkspace(debug = DebugLevel.KeepFiles)
        data_set = range(1, self.number_sample + 1)
        data_set = [str(i) for i in data_set]
        # data_set = ["1","2","3","4","5","6","7","8","9","10"]
        y_param = dict(zip(data_set, y))
        y = np.matrix(y)
        inner_product_y = np.dot(y.T,y)
        K_y_param = {}

        Kernel_param = {}
        inner_product = np.dot(X, np.transpose(X) )
        for i in data_set:
            for j in data_set:
                Kernel_param[(i,j)] = inner_product[int(i)-1,int(j)-1]
                K_y_param[(i,j)] = inner_product_y[int(i)-1,int(j)-1]
        db = ws.add_database()

        i = GamsSet(db, "i", 1, "number of data samples")
        for d in data_set:
            i.add_record(d)

        K_param_db = GamsParameter(db, "K_x", 2,"kernel or inner product")
        for k, v in Kernel_param.iteritems():
            K_param_db.add_record(k).value = v

        K_y_param_db = GamsParameter(db, "K_y", 2, "y kernel" )
        for k, v in K_y_param.iteritems():
            K_y_param_db.add_record(k).value = v

        y_param_db = GamsParameter(db, "y",1, "target label")
        for d in data_set:
            y_param_db.add_record(d).value = y_param[d]

        C_db = GamsParameter(db, "C", 0, "regularization parameter")
        C_db.add_record().value = self.C

        t = GamsJob(ws, source = self._get_model_text())
        opt = GamsOptions(ws)
        opt.all_model_types = "baron"
        opt.defines["gdxincname"] = db.name
        t.run(opt, databases=db)
        ################################### GAMS solving QCP done ####################################################
        # Retrieve solution from output database
        for alpha in t.out_db["alpha"]:
            self.dual_coef_.append(alpha.level)
        # round solutions for numerical computations, e.g., returned 1.99999 from GAMS but it should be 2
        self.dual_coef_ = np.round(self.dual_coef_, decimals= 8)
        print self.dual_coef_[self.dual_coef_>0]
        temp = np.dot(np.asmatrix(np.multiply(y, self.dual_coef_)), X)
        self.coef_ = np.sum(temp, axis=0)

        self.support = np.where(np.logical_and((0< self.dual_coef_), (self.dual_coef_<self.C)))[0]
        self.support_vectors_ = X[self.support,:]

        # indices of data sample used to comput beta_0
        intercept_ind = self.support[0]
        # y was casted into matrix
        self.intercept = 1/y[0,intercept_ind] - np.inner(X[intercept_ind,:], self.coef_)

        # check the support vector when dual_coef == self.C
        # temp_ind = np.where(self.dual_coef_==self.C)[0]
        # digamamma = 1- np.multiply(y[0,temp_ind],(np.dot(X[temp_ind,:], self.coef_.T) +self.intercept))
        # digamamma = np.ravel(digamamma)
        #
        # self.support = np.append(self.support,temp_ind[digamamma==0])
        non_margin_ind = np.where(self.dual_coef_ == self.C)[0]
        self.non_margin_support_vectors = X[non_margin_ind,:]
        self.n_support_.append(np.sum(y[0,self.support] == -1)) # number of class -1 support vectors
        self.n_support_.append(np.sum(y[0,self.support] == 1)) # number of class 1 support vectors

    def predict(self, X):
        """

        :param X: numpy array, [number of sample, NO of features], feature already scaled
        :return: array of -1 or 1
        """
        def predict_single(a):
            temp = np.inner(a, self.coef_) + self.intercept
            label = 1 if temp >= 0 else -1
            return label
        return np.apply_along_axis(predict_single,1, X)

def plot_decision_regions(X,y, classifier, test_idx = None, resolution = 0.01):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha = 0.5, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha = 0.8, c=cmap(idx),marker=markers[idx],label=cl)
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1], c='', alpha = 1.0, linewidths=1, marker ='o', s=55, label='test set')
    plt.scatter(classifier.support_vectors_[:,0], classifier.support_vectors_[:, 1], c='', alpha=1.0, linewidths=1, marker='o', s=55, label='margin support vector')

    plt.show()


if __name__ == "__main__":
    X, y = make_classification(n_samples = 50, n_features = 2,  n_redundant=0, n_classes=2, random_state=1)
    y[y==0] = -1

    svm_cl = svm(C = 10)
    svm_cl.fit(X, y, solve_dual = True)
    plot_decision_regions(X,y,svm_cl)