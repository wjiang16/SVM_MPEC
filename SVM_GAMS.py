# This script implements soft margin Support vector machine classifier for two classes
# using GAMS as the optimization solver
# Wei Jiang
# 11/26/2016

import numpy as np
from gams import *
import os
import sys
from sklearn.datasets import make_classification

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

    def __init__(self, C = 1 ):
        self.C = C
        self.support = []
        self.support_vectors_ = None
        self.n_support_ = []
        self.dual_coef_ = []
        self.coef_ = []
        self.intercept = None

    def _get_model_text(self):
        return '''
        Sets
            i number of data samples
        ;
        Alias(i,j);
        Scalar C regularization parameter;

        Parameters
            y(i)
            K_y(i,j) y_i times y_j
            K_x(i,j) inner product or kernel of vector x_i and x_j;

        Positive Variable
            alpha(i) dual variable in the primal problem
        Variable
            z  lagrangian of the dual problem
        Equations
            obj     objective function
            dual_cap(i)
            cons_equal;

        obj ..  z =e= sum(i, alpha(i)) - 0.5 * sum(i, sum(j, alpha(i)*alpha(j)*K_y(i,j)*K_x(i,j) ));
        dual_cap(i) ..  alpha(i) =l= C;
        cons_equal ..       sum(i, alpha(i) * y(i)) =e= 0;

        Model svm_dual /all/;
        Solve svm_dual using QCP maximizing z;
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

        i = db.add_set("i", 1)
        # j = db.add_set("j", 1)
        for d in data_set:
            i.add_record(d)
            # j.add_record(d)

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
        t.run(databases = db)

        # Retrieve solution from output database
        for alpha in t.out_db["alpha"]:
            self.dual_coef_.append(alpha.level)


    # def predict(self, x, y):

if __name__ == "__main__":
    X, y = make_classification(n_samples = 10, n_features = 2, n_redundant=0, n_classes=2)
    y[y==0] = -1
    print X, y
    svm_cl = svm()
    svm_cl.fit(X, y, solve_dual = True)
    print svm_cl.dual_coef_