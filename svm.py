# This script implements soft margin Support vector machine classifier for two classes
# using GAMS as the optimization solver
# Wei Jiang
# 11/26/2016

import numpy as np
import pandas as pd
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
cons_equal ..   sum(i, alpha(i) * y(i)) =e= 0;

Model svm_dual /all/;
Solve svm_dual using QCP maximizing L;
'''



    def _add_data_db(self, X, y):
        # Helper method, add data to database
        self.number_sample = X.shape[0]
        ws = GamsWorkspace(debug = DebugLevel.KeepFiles)
        data_set = range(1, self.number_sample + 1)
        data_set = [str(i) for i in data_set]
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
        return db, ws

    def _get_support_ind(self,y):
        # helper function to get the indices for the support vectors
        self.support = np.where(np.logical_and((0 < self.dual_coef_), (self.dual_coef_ < self.C)))[0]

    def fit(self, X, y):
        """
        X: numpy array, [number of sample, NO of features], feature already scaled
        y: array of -1 or 1
        solve_dual: boolean, if true, solve the quadratic dual problem
                    otherwise, solve the primal problem
        """

        db, ws = self._add_data_db(X,y)


        t = GamsJob(ws, source = self._get_model_text())
        opt = GamsOptions(ws)
        opt.all_model_types = "conopt"
        opt.defines["gdxincname"] = db.name
        t.run(opt, databases=db)
        ################################### GAMS solving QCP done ####################################################
        # Retrieve solution from output database
        for alpha in t.out_db["alpha"]:
            self.dual_coef_.append(alpha.level)
        # round solutions for numerical computations, e.g., returned 1.99999 from GAMS but it should be 2
        self.dual_coef_ = np.round(self.dual_coef_, decimals= 8)

        # print self.dual_coef_[self.dual_coef_>0]
        self.coef_ = np.dot(np.asmatrix(np.multiply(y, self.dual_coef_)), X)
        # self.coef_ = np.sum(temp, axis=0)
        self._get_support_ind(y)
        # print self.support
        # print self.dual_coef_[self.support]
        # print y[self.support]

        self.support_vectors_ = X[self.support,:]

        # if there is no support margin vector, then parameter C is too small for the problem
        # warn the user to provide a larger C

        # indices of data sample used to comput beta_0
        try:
            intercept_ind = self.support[0]
        except:
            print('Parameter C too small for your data set, please provide a larger C')
            raise
        # for i in intercept_ind:
        self.intercept = 1/y[intercept_ind] - np.inner(X[intercept_ind,:], self.coef_)

        non_margin_ind = np.where(self.dual_coef_ == self.C)[0]
        self.non_margin_support_vectors = X[non_margin_ind,:]
        self.n_support_.append(np.sum(y[self.support] == -1)) # number of class -1 support vectors
        self.n_support_.append(np.sum(y[self.support] == 1)) # number of class 1 support vectors

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


class svm_imblanced(svm):

    def __init__(self, C_pos, C_neg):
        # C_pos regularization parameter for positive classes
        # C_neg regularization parameter for negative classes
        svm.__init__(self, C_pos)
        self.C_neg = C_neg

    def _get_model_text(self):
        # SVM model for imbalanced data
        return '''
Sets
i number of data samples
j_pos(i) number of positive samples
j_neg(i) number of negative samples
;
Alias(i,z);
Parameters
y(i) target label
K_y(i,z) y kernel
K_x(i,z) kernel or inner product
;
Scalar
C regularization parameter for positive samples
C_neg regularization parameter for negative samples
;
$if not set gdxincname $abort 'no include file name for data file provided'
$GDXin %gdxincname%
$load i j_pos j_neg y K_y K_x C C_neg
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
dual_cap_pos(j_pos)
dual_cap_neg(j_neg)
cons_equal;

obj ..  L =e= sum(i, alpha(i)) - 0.5 * sum(i, sum(z, alpha(i)*alpha(z)*K_y(i,z)*K_x(i,z) ));
dual_cap_pos(j_pos) ..  alpha(j_pos) =l= C;
dual_cap_neg(j_neg) ..  alpha(j_neg) =l= C_neg;
cons_equal ..       sum(i, alpha(i) * y(i)) =e= 0;

Model svm_dual /all/;
Solve svm_dual using QCP maximizing L;
'''
    def _add_data_db(self, X, y):
        db, ws = svm._add_data_db(self,X,y)
        j_pos = np.where(y==1)[0]
        j_neg = np.where(y==-1)[0]
        j_pos = [str(i+1) for i in j_pos]
        j_neg = [str(i+1) for i in j_neg]

        i = GamsSet(db, "j_pos", 1, "number of positive samples")
        for d in j_pos:
            i.add_record(d)
        j = GamsSet(db, "j_neg", 1, "number of negative samples")
        for d in j_neg:
            j.add_record(d)

        C_db = GamsParameter(db, "C_neg", 0, "regularization parameter for negative samples")
        C_db.add_record().value = self.C_neg
        return db, ws

    def _get_support_ind(self, y):
        self.support = np.where(np.logical_or(np.logical_and(np.logical_and((0 < self.dual_coef_), (self.dual_coef_ < self.C))
                                , y == 1), np.logical_and(np.logical_and((0 < self.dual_coef_), (self.dual_coef_ < self.C_neg))
                                , y == -1))
                                )[0]

def plot_decision_regions(X,y, classifier, test_idx = None, resolution = 0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha = 0.8, c=cmap(idx),marker=markers[idx],label=cl)
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1], c='', alpha = 1.0, linewidths=1, marker ='o', s=55, label='test set')
    plt.scatter(classifier.support_vectors_[:,0], classifier.support_vectors_[:, 1], c='', alpha=1.0, linewidths=1, marker='o', s=55, label='margin support vector')
    # plt.legend(loc='best')
    plt.title('SVM classification')
    plt.show()


if __name__ == "__main__":
    # import cProfile, pstats
    #
    # pr = cProfile.Profile()
    # pr.enable()
    X, y = make_classification(n_samples = 500, n_features = 2, n_redundant=0, n_classes=2, random_state=1)
    # X, y = make_classification(n_samples=500, n_features=2, weights= [0.1,0.9], n_redundant=0, n_classes=2, random_state=1)
    y[y==0] = -1
    print y[y==1].shape
    # df = pd.DataFrame(X)
    # df['Y'] = y
    # df.to_csv('random_data_classification.csv')

    # svm_cl = svm_imblanced(C_pos = 10, C_neg = 1000)
    # svm_cl.fit(X, y)
    # plot_decision_regions(X,y,svm_cl)

    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('tottime')
    # ps.print_stats(15)

    # svm_cl = svm(C = 1)
    # svm_cl.fit(X, y)
    # plot_decision_regions(X, y, svm_cl)

    svm_cl = svm(C=0.01)
    svm_cl.fit(X, y)
    plot_decision_regions(X, y, svm_cl)