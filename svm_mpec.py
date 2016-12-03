# This script solve the optimization problem of optimizing hyperparameter for soft margin SVM (two classes) using
# cross-validation as MPEC using GAMS as the optimization solver
# Wei Jiang
# 11/27/2016

import numpy as np
from gams import *
import os
import sys
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
class svm_mpec():
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

    def __init__(self, ):
        self.C = None
        self.support = []
        self.support_vectors_ = None
        self.non_margin_support_vectors = None
        self.n_support_ = []
        self.dual_coef_ = []
        self.coef_ = []
        self.intercept = None

    def _get_model_text_cp(self):
        return '''
Sets
i number of data samples
cv_1(i) number of sample fold 1
cv_2(i) number of sample fold 2
;
Alias(i,z);
Alias(cv_1, z1);
Alias(cv_2, z2);
Parameters
y(i) target label
K_y(i,z) y kernel
K_x(i,z) kernel or inner product
;

$if not set gdxincname $abort 'no include file name for data file provided'
$GDXin %gdxincname%
$load i cv_1 cv_2 y K_y K_x
$GDXin
;

Parameter number;
Number = Card(cv_1);


Positive Variable
alpha_cv_1(cv_1)
alpha_cv_2(cv_2)
xi_cv_1(cv_1)     soft margin violation term
xi_cv_2(cv_2)
B_cv_1(cv_1)
B_cv_2(cv_2)
lambda_cv_1(cv_1)
lambda_cv_2(cv_2)
;
Variable
C
beta_cv_1   intercept term for the separating hyperplane
beta_cv_2
L       cross-validation error using accuracy measure
;
Equations
obj     objective function
test_cons1_cv_1(cv_1)
test_cons1_cv_2(cv_2)
test_cons2_cv_1(cv_1)
test_cons2_cv_2(cv_2)

lower_cons1_cv_1(cv_1)
lower_cons1_cv_2(cv_2)
lower_cons2_cv_1(cv_1)
lower_cons2_cv_2(cv_2)
lower_cons3_cv_1
lower_cons3_cv_2
;

obj ..  L =e= 0.5*(sum(cv_1, B_cv_1(cv_1))/card(cv_1) + sum(cv_2, B_cv_2(cv_2))/card(cv_2)) ;
test_cons1_cv_1(cv_1) .. sum(cv_2, alpha_cv_2(cv_2) * K_y(cv_1, cv_2) * K_x(cv_1,cv_2)) + y(cv_1) * beta_cv_2 + lambda_cv_1(cv_1) =g= 0;
test_cons1_cv_2(cv_2) .. sum(cv_1, alpha_cv_1(cv_1) * K_y(cv_2, cv_1) * K_x(cv_2,cv_1)) + y(cv_2) * beta_cv_1 + lambda_cv_2(cv_2) =g= 0;
test_cons2_cv_1(cv_1) .. 1 - B_cv_1(cv_1) =g= 0;
test_cons2_cv_2(cv_2) .. 1 - B_cv_2(cv_2) =g= 0;

lower_cons1_cv_1(cv_1) .. C - alpha_cv_1(cv_1) =g= 0;
lower_cons1_cv_2(cv_2) .. C - alpha_cv_2(cv_2) =g= 0;
lower_cons2_cv_1(cv_1) .. sum(z1, alpha_cv_1(z1) * K_y(cv_1, z1) * K_x(cv_1,z1)) + y(cv_1) * beta_cv_1 - (1-xi_cv_1(cv_1)) =g= 0;
lower_cons2_cv_2(cv_2) .. sum(z2, alpha_cv_2(z2) * K_y(cv_2, z2) * K_x(cv_2,z2)) + y(cv_2) * beta_cv_2 - (1-xi_cv_2(cv_2)) =g= 0;
lower_cons3_cv_1 .. sum(cv_1,alpha_cv_1(cv_1) * y(cv_1) ) =e= 0;
lower_cons3_cv_2 .. sum(cv_2,alpha_cv_2(cv_2) * y(cv_2) ) =e= 0;

Model svm_mpec /obj, test_cons1_cv_1.B_cv_1,test_cons1_cv_2.B_cv_2, test_cons2_cv_1.lambda_cv_1, test_cons2_cv_2.lambda_cv_2,
lower_cons1_cv_1.xi_cv_1, lower_cons1_cv_2.xi_cv_2, lower_cons2_cv_1.alpha_cv_1,lower_cons2_cv_2.alpha_cv_2,
lower_cons3_cv_1, lower_cons3_cv_2/;
C.l = 200
*option MPEC=nlpec
Solve svm_mpec using mpec minimizing L;
Display Number
'''

    def _get_model_text_nlp(self):
        return '''
Sets
i number of data samples
cv_1(i) number of sample fold 1
cv_2(i) number of sample fold 2
;
Alias(i,z);
Alias(cv_1, z1);
Alias(cv_2, z2);
Parameters
y(i) target label
K_y(i,z) y kernel
K_x(i,z) kernel or inner product
;

$if not set gdxincname $abort 'no include file name for data file provided'
$GDXin %gdxincname%
$load i cv_1 cv_2 y K_y K_x
$GDXin
;

Parameter number;
Number = Card(cv_1);

Positive Variable
alpha_cv_1(cv_1)
alpha_cv_2(cv_2)
xi_cv_1(cv_1)     soft margin violation term
xi_cv_2(cv_2)
B_cv_1(cv_1)
B_cv_2(cv_2)
lambda_cv_1(cv_1)
lambda_cv_2(cv_2)
gap1_cv_1
gap2_cv_1
gap1_cv_2
gap2_cv_2
;
Variable
C
beta_cv_1   intercept term for the separating hyperplane
beta_cv_2
L       cross-validation error using accuracy measure
;
Equations
obj     objective function
test_duality_cv_1
test_duality_cv_2
dual_cons1_cv_1(cv_1)
dual_cons1_cv_2(cv_2)
test_cons2_cv_1(cv_1)
test_cons2_cv_2(cv_2)
train_duality_cv_1
train_duailty_cv_2

lower_cons1_cv_1(cv_1)
lower_cons1_cv_2(cv_2)
lower_cons2_cv_1(cv_1)
lower_cons2_cv_2(cv_2)
lower_cons3_cv_1
lower_cons3_cv_2
;

obj ..  L =e= 0.5*(sum(cv_1, B_cv_1(cv_1))/card(cv_1) + sum(cv_2, B_cv_2(cv_2))/card(cv_2)) + gap1_cv_1 + gap1_cv_2 + gap2_cv_1 + gap2_cv_2 ;
test_duality_cv_1 .. gap1_cv_1 =e= sum(cv_1, B_cv_1(cv_1)*(sum(cv_2, alpha_cv_2(cv_2) * K_y(cv_1, cv_2) * K_x(cv_1,cv_2)) + y(cv_1) * beta_cv_2) + lambda_cv_1(cv_1));
test_duality_cv_2 .. gap1_cv_2 =e= sum(cv_2, B_cv_2(cv_2)*(sum(cv_1, alpha_cv_1(cv_1) * K_y(cv_1, cv_2) * K_x(cv_1,cv_2)) + y(cv_2) * beta_cv_1) + lambda_cv_2(cv_2));
dual_cons1_cv_1(cv_1) .. sum(cv_2, alpha_cv_2(cv_2) * K_y(cv_1, cv_2) * K_x(cv_1,cv_2)) + y(cv_1) * beta_cv_2 + lambda_cv_1(cv_1) =g= 0;
dual_cons1_cv_2(cv_2) .. sum(cv_1, alpha_cv_1(cv_1) * K_y(cv_2, cv_1) * K_x(cv_2,cv_1)) + y(cv_2) * beta_cv_1 + lambda_cv_2(cv_2) =g= 0;

*test_cons1_cv_2(cv_2) .. sum(cv_1, alpha_cv_1(cv_1) * K_y(cv_2, cv_1) * K_x(cv_2,cv_1)) + y(cv_2) * beta_cv_1 + lambda_cv_2(cv_2) =g= 0;
test_cons2_cv_1(cv_1) .. 1 - B_cv_1(cv_1) =g= 0;
test_cons2_cv_2(cv_2) .. 1 - B_cv_2(cv_2) =g= 0;

train_duality_cv_1 .. -gap2_cv_1 =e= sum(cv_1, alpha_cv_1(cv_1)) -  sum(cv_1, sum(z1, alpha_cv_1(cv_1) * alpha_cv_1(z1) * K_y(cv_1, z1) * K_x(cv_1,z1))) - C * sum(cv_1,xi_cv_1(cv_1));
train_duailty_cv_2 .. -gap2_cv_2 =e= sum(cv_2, alpha_cv_2(cv_2)) -  sum(cv_2, sum(z2, alpha_cv_2(cv_2) * alpha_cv_2(z2) * K_y(cv_2, z2) * K_x(cv_2,z2))) - C * sum(cv_2,xi_cv_2(cv_2)) ;

lower_cons1_cv_1(cv_1) .. C - alpha_cv_1(cv_1) =g= 0;
lower_cons1_cv_2(cv_2) .. C - alpha_cv_2(cv_2) =g= 0;
lower_cons2_cv_1(cv_1) .. sum(z1, alpha_cv_1(z1) * K_y(cv_1, z1) * K_x(cv_1,z1)) + y(cv_1) * beta_cv_1 - (1-xi_cv_1(cv_1)) =g= 0;
lower_cons2_cv_2(cv_2) .. sum(z2, alpha_cv_2(z2) * K_y(cv_2, z2) * K_x(cv_2,z2)) + y(cv_2) * beta_cv_2 - (1-xi_cv_2(cv_2)) =g= 0;
lower_cons3_cv_1 .. sum(cv_1,alpha_cv_1(cv_1) * y(cv_1) ) =e= 0;
lower_cons3_cv_2 .. sum(cv_2,alpha_cv_2(cv_2) * y(cv_2) ) =e= 0;
C.l = 100;
alpha_cv_1.l(cv_1) = 2;
alpha_cv_2.l(cv_2) = 2;
Model svm_bilevel /all/;

*option MPEC=nlpec
Solve svm_bilevel using nlp minimizing L;
Display C.l
'''
    def _get_model_text_nlp_cv5(self):
        return '''
Sets
i number of data samples
cv_1(i) number of sample fold cv_1 test
cv_11(i) number of sample fold cv_1 train
cv_2(i) number of sample fold cv_2 test
cv_21(i) number of sample fold cv_2 train
cv_3(i) number of sample fold cv_3 test
cv_31(i) number of sample fold cv_3 train
cv_4(i) number of sample fold cv_4 test
cv_41(i) number of sample fold cv_4 train
cv_5(i) number of sample fold cv_5 test
cv_51(i) number of sample fold cv_5 train
;
Alias(i,z);
Alias(cv_11, z1);
Alias(cv_21, z2);
Alias(cv_31, z3);
Alias(cv_41, z4);
Alias(cv_51, z5);
Parameters
y(i) target label
K_y(i,z) y kernel
K_x(i,z) kernel or inner product
;

$if not set gdxincname $abort 'no include file name for data file provided'
$GDXin %gdxincname%
$load i cv_1 cv_11 cv_2 cv_21 cv_3 cv_31 cv_4 cv_41 cv_5 cv_51 y K_y K_x
$GDXin
;
Positive Variable
B_cv_1(cv_1)
lambda_cv_1(cv_1)
gap_cv_1

alpha_cv_11(cv_11)
xi_cv_11(cv_11)
gap_cv_11

B_cv_2(cv_2)
lambda_cv_2(cv_2)
gap_cv_2

alpha_cv_21(cv_21)
xi_cv_21(cv_21)
gap_cv_21

B_cv_3(cv_3)
lambda_cv_3(cv_3)
gap_cv_3

alpha_cv_31(cv_31)
xi_cv_31(cv_31)
gap_cv_31

B_cv_4(cv_4)
lambda_cv_4(cv_4)
gap_cv_4

alpha_cv_41(cv_41)
xi_cv_41(cv_41)
gap_cv_41

B_cv_5(cv_5)
lambda_cv_5(cv_5)
gap_cv_5

alpha_cv_51(cv_51)
xi_cv_51(cv_51)
gap_cv_51
;
Variable
C
beta_cv_11   intercept term for the separating hyperplane
beta_cv_21
beta_cv_31
beta_cv_41
beta_cv_51
L       cross-validation error using accuracy measure
;
Equations
obj
test_duality_cv_1
dual_cons1_cv_1(cv_1)
test_cons2_cv_1(cv_1)
train_duailty_cv_11
lower_cons1_cv_11(cv_11)
lower_cons2_cv_11(cv_11)
lower_cons3_cv_11

test_duality_cv_2
dual_cons1_cv_2(cv_2)
test_cons2_cv_2(cv_2)
train_duailty_cv_21
lower_cons1_cv_21(cv_21)
lower_cons2_cv_21(cv_21)
lower_cons3_cv_21

test_duality_cv_3
dual_cons1_cv_3(cv_3)
test_cons2_cv_3(cv_3)
train_duailty_cv_31
lower_cons1_cv_31(cv_31)
lower_cons2_cv_31(cv_31)
lower_cons3_cv_31

test_duality_cv_4
dual_cons1_cv_4(cv_4)
test_cons2_cv_4(cv_4)
train_duailty_cv_41
lower_cons1_cv_41(cv_41)
lower_cons2_cv_41(cv_41)
lower_cons3_cv_41

test_duality_cv_5
dual_cons1_cv_5(cv_5)
test_cons2_cv_5(cv_5)
train_duailty_cv_51
lower_cons1_cv_51(cv_51)
lower_cons2_cv_51(cv_51)
lower_cons3_cv_51
;
obj ..  L =e= 0.2*(sum(cv_1, B_cv_1(cv_1))/card(cv_1) + sum(cv_2, B_cv_2(cv_2))/card(cv_2) + sum(cv_3, B_cv_3(cv_3))/card(cv_3)
                + sum(cv_4, B_cv_4(cv_4))/card(cv_4) + sum(cv_5, B_cv_5(cv_5))/card(cv_5)
        ) + gap_cv_1 + 100*gap_cv_11 + gap_cv_2 + 100*gap_cv_21 + gap_cv_3 + 100*gap_cv_31 +gap_cv_4 +100*gap_cv_41 + gap_cv_5 + 100*gap_cv_51 ;
test_duality_cv_1 .. gap_cv_1 =e= sum(cv_1, B_cv_1(cv_1)*(sum(cv_11, alpha_cv_11(cv_11) * K_y(cv_1, cv_11) * K_x(cv_1,cv_11)) + y(cv_1) * beta_cv_11) + lambda_cv_1(cv_1));
dual_cons1_cv_1(cv_1) .. sum(cv_11, alpha_cv_11(cv_11) * K_y(cv_1, cv_11) * K_x(cv_1,cv_11)) + y(cv_1) * beta_cv_11 + lambda_cv_1(cv_1) =g= 0;
test_cons2_cv_1(cv_1) .. 1 - B_cv_1(cv_1) =g= 0;

train_duailty_cv_11 .. -gap_cv_11 =e= sum(cv_11, alpha_cv_11(cv_11)) -  sum(cv_11, sum(z1, alpha_cv_11(cv_11) * alpha_cv_11(z1) * K_y(cv_11, z1) * K_x(cv_11,z1))) - C * sum(cv_11,xi_cv_11(cv_11)) ;
lower_cons1_cv_11(cv_11) .. C - alpha_cv_11(cv_11) =g= 0;
lower_cons2_cv_11(cv_11) .. sum(z1, alpha_cv_11(z1) * K_y(cv_11, z1) * K_x(cv_11,z1)) + y(cv_11) * beta_cv_11 - (1-xi_cv_11(cv_11)) =g= 0;
lower_cons3_cv_11 .. sum(cv_11,alpha_cv_11(cv_11) * y(cv_11) ) =e= 0;

test_duality_cv_2 .. gap_cv_2 =e= sum(cv_2, B_cv_2(cv_2)*(sum(cv_21, alpha_cv_21(cv_21) * K_y(cv_2, cv_21) * K_x(cv_2,cv_21)) + y(cv_2) * beta_cv_21) + lambda_cv_2(cv_2));
dual_cons1_cv_2(cv_2) .. sum(cv_21, alpha_cv_21(cv_21) * K_y(cv_2, cv_21) * K_x(cv_2,cv_21)) + y(cv_2) * beta_cv_21 + lambda_cv_2(cv_2) =g= 0;
test_cons2_cv_2(cv_2) .. 1 - B_cv_2(cv_2) =g= 0;

train_duailty_cv_21 .. -gap_cv_21 =e= sum(cv_21, alpha_cv_21(cv_21)) -  sum(cv_21, sum(z2, alpha_cv_21(cv_21) * alpha_cv_21(z2) * K_y(cv_21, z2) * K_x(cv_21,z2))) - C * sum(cv_21,xi_cv_21(cv_21)) ;
lower_cons1_cv_21(cv_21) .. C - alpha_cv_21(cv_21) =g= 0;
lower_cons2_cv_21(cv_21) .. sum(z2, alpha_cv_21(z2) * K_y(cv_21, z2) * K_x(cv_21,z2)) + y(cv_21) * beta_cv_21 - (1-xi_cv_21(cv_21)) =g= 0;
lower_cons3_cv_21 .. sum(cv_21,alpha_cv_21(cv_21) * y(cv_21) ) =e= 0;

test_duality_cv_3 .. gap_cv_3 =e= sum(cv_3, B_cv_3(cv_3)*(sum(cv_31, alpha_cv_31(cv_31) * K_y(cv_3, cv_31) * K_x(cv_3,cv_31)) + y(cv_3) * beta_cv_31) + lambda_cv_3(cv_3));
dual_cons1_cv_3(cv_3) .. sum(cv_31, alpha_cv_31(cv_31) * K_y(cv_3, cv_31) * K_x(cv_3,cv_31)) + y(cv_3) * beta_cv_31 + lambda_cv_3(cv_3) =g= 0;
test_cons2_cv_3(cv_3) .. 1 - B_cv_3(cv_3) =g= 0;

train_duailty_cv_31 .. -gap_cv_31 =e= sum(cv_31, alpha_cv_31(cv_31)) -  sum(cv_31, sum(z3, alpha_cv_31(cv_31) * alpha_cv_31(z3) * K_y(cv_31, z3) * K_x(cv_31,z3))) - C * sum(cv_31,xi_cv_31(cv_31)) ;
lower_cons1_cv_31(cv_31) .. C - alpha_cv_31(cv_31) =g= 0;
lower_cons2_cv_31(cv_31) .. sum(z3, alpha_cv_31(z3) * K_y(cv_31, z3) * K_x(cv_31,z3)) + y(cv_31) * beta_cv_31 - (1-xi_cv_31(cv_31)) =g= 0;
lower_cons3_cv_31 .. sum(cv_31,alpha_cv_31(cv_31) * y(cv_31) ) =e= 0;

test_duality_cv_4 .. gap_cv_4 =e= sum(cv_4, B_cv_4(cv_4)*(sum(cv_41, alpha_cv_41(cv_41) * K_y(cv_4, cv_41) * K_x(cv_4,cv_41)) + y(cv_4) * beta_cv_41) + lambda_cv_4(cv_4));
dual_cons1_cv_4(cv_4) .. sum(cv_41, alpha_cv_41(cv_41) * K_y(cv_4, cv_41) * K_x(cv_4,cv_41)) + y(cv_4) * beta_cv_41 + lambda_cv_4(cv_4) =g= 0;
test_cons2_cv_4(cv_4) .. 1 - B_cv_4(cv_4) =g= 0;

train_duailty_cv_41 .. -gap_cv_41 =e= sum(cv_41, alpha_cv_41(cv_41)) -  sum(cv_41, sum(z4, alpha_cv_41(cv_41) * alpha_cv_41(z4) * K_y(cv_41, z4) * K_x(cv_41,z4))) - C * sum(cv_41,xi_cv_41(cv_41)) ;
lower_cons1_cv_41(cv_41) .. C - alpha_cv_41(cv_41) =g= 0;
lower_cons2_cv_41(cv_41) .. sum(z4, alpha_cv_41(z4) * K_y(cv_41, z4) * K_x(cv_41,z4)) + y(cv_41) * beta_cv_41 - (1-xi_cv_41(cv_41)) =g= 0;
lower_cons3_cv_41 .. sum(cv_41,alpha_cv_41(cv_41) * y(cv_41) ) =e= 0;

test_duality_cv_5 .. gap_cv_5 =e= sum(cv_5, B_cv_5(cv_5)*(sum(cv_51, alpha_cv_51(cv_51) * K_y(cv_5, cv_51) * K_x(cv_5,cv_51)) + y(cv_5) * beta_cv_51) + lambda_cv_5(cv_5));
dual_cons1_cv_5(cv_5) .. sum(cv_51, alpha_cv_51(cv_51) * K_y(cv_5, cv_51) * K_x(cv_5,cv_51)) + y(cv_5) * beta_cv_51 + lambda_cv_5(cv_5) =g= 0;
test_cons2_cv_5(cv_5) .. 1 - B_cv_5(cv_5) =g= 0;

train_duailty_cv_51 .. -gap_cv_51 =e= sum(cv_51, alpha_cv_51(cv_51)) -  sum(cv_51, sum(z5, alpha_cv_51(cv_51) * alpha_cv_51(z5) * K_y(cv_51, z5) * K_x(cv_51,z5))) - C * sum(cv_51,xi_cv_51(cv_51)) ;
lower_cons1_cv_51(cv_51) .. C - alpha_cv_51(cv_51) =g= 0;
lower_cons2_cv_51(cv_51) .. sum(z5, alpha_cv_51(z5) * K_y(cv_51, z5) * K_x(cv_51,z5)) + y(cv_51) * beta_cv_51 - (1-xi_cv_51(cv_51)) =g= 0;
lower_cons3_cv_51 .. sum(cv_51,alpha_cv_51(cv_51) * y(cv_51) ) =e= 0;

C.l = 30;
alpha_cv_11.l(cv_11) = 2;
alpha_cv_21.l(cv_21) = 2;
alpha_cv_31.l(cv_31) = 2;
alpha_cv_41.l(cv_41) = 2;
alpha_cv_51.l(cv_51) = 2;
Model svm_bilevel /all/;

*option MPEC=nlpec
Solve svm_bilevel using nlp minimizing L;
Display L.l, C.l, gap_cv_1.l, gap_cv_11.l, gap_cv_2.l, gap_cv_21.l, gap_cv_3.l, gap_cv_31.l,gap_cv_4.l, gap_cv_41.l, gap_cv_5.l, gap_cv_51.l
        '''

    def fit(self, X, y, cv = 2):
        """
        X: numpy array, [number of sample, NO of features], feature already scaled
        y: array of -1 or 1
        solve_dual: boolean, if true, solve the quadratic dual problem
                    otherwise, solve the primal problem
        """
        ########## Split data into k fold for cross-validation ##########
        skf = StratifiedKFold(n_splits= cv)
        # indices of training data for each splitting a list of array
        cv_train_indices =[]
        cv_test_indices = []
        for train_index, test_index in skf.split(X,y):
            train_index = [str(i+1) for i in train_index]
            test_index = [str(i+1) for i in test_index]
            cv_train_indices.append(train_index)
            cv_test_indices.append(test_index)
        ########## Data splitting done  ################################

        self.number_sample = X.shape[0]
        ws = GamsWorkspace(debug = DebugLevel.KeepFiles)
        data_set = range(1, self.number_sample + 1)
        data_set = [str(i) for i in data_set]
        # cv_train = [str(i+1) for i in cv_train_indices[0]]
        # cv_test = [str(i+1) for i in cv_test_indices[0]]
        # print cv_train, cv_test
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

        def add_set(set_name, data):
            i = GamsSet(db, set_name, 1, "number of data samples")
            for d in data:
                i.add_record(d)

        add_set("i", data_set)
        test_set = ['cv_1','cv_2','cv_3','cv_4','cv_5']
        train_set = ['cv_11','cv_21','cv_31','cv_41','cv_51']
        for i in range(0, cv):
            add_set(test_set[i], cv_test_indices[i])
            add_set(train_set[i], cv_train_indices[i])

        K_param_db = GamsParameter(db, "K_x", 2,"kernel or inner product")
        for k, v in Kernel_param.iteritems():
            K_param_db.add_record(k).value = v

        K_y_param_db = GamsParameter(db, "K_y", 2, "y kernel" )
        for k, v in K_y_param.iteritems():
            K_y_param_db.add_record(k).value = v

        y_param_db = GamsParameter(db, "y",1, "target label")
        for d in data_set:
            y_param_db.add_record(d).value = y_param[d]

        # C_db = GamsParameter(db, "C", 0, "regularization parameter")
        # C_db.add_record().value = self.C

        t = GamsJob(ws, source = self._get_model_text_nlp_cv5())
        opt = GamsOptions(ws)
        opt.all_model_types = "conopt"
        opt.defines["gdxincname"] = db.name
        t.run(opt, databases=db)
        ################################### GAMS solving QCP done ####################################################
        # Retrieve solution from output database
        for c in t.out_db["C"]:
            self.C = c.level
        # # round solutions for numerical computations, e.g., returned 1.99999 from GAMS but it should be 2
        # self.dual_coef_ = np.round(self.dual_coef_, decimals= 8)
        # # print self.dual_coef_[self.dual_coef_>0]
        # temp = np.dot(np.asmatrix(np.multiply(y, self.dual_coef_)), X)
        # self.coef_ = np.sum(temp, axis=0)
        #
        # self.support = np.where(np.logical_and((0< self.dual_coef_), (self.dual_coef_<self.C)))[0]
        # self.support_vectors_ = X[self.support,:]
        #
        # # indices of data sample used to comput beta_0
        # intercept_ind = self.support[0]
        # # y was casted into matrix
        # self.intercept = 1/y[0,intercept_ind] - np.inner(X[intercept_ind,:], self.coef_)
        #
        # # check the support vector when dual_coef == self.C
        # # temp_ind = np.where(self.dual_coef_==self.C)[0]
        # # digamamma = 1- np.multiply(y[0,temp_ind],(np.dot(X[temp_ind,:], self.coef_.T) +self.intercept))
        # # digamamma = np.ravel(digamamma)
        # #
        # # self.support = np.append(self.support,temp_ind[digamamma==0])
        # non_margin_ind = np.where(self.dual_coef_ == self.C)[0]
        # self.non_margin_support_vectors = X[non_margin_ind,:]
        # self.n_support_.append(np.sum(y[0,self.support] == -1)) # number of class -1 support vectors
        # self.n_support_.append(np.sum(y[0,self.support] == 1)) # number of class 1 support vectors

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
    plt.show()


if __name__ == "__main__":
    X, y = make_classification(n_samples = 500, n_features = 2,  n_redundant=0, n_classes=2, random_state=1)
    y[y==0] = -1
    print X.shape

    svm_cl = svm_mpec()
    svm_cl.fit(X, y, cv=5)
    print svm_cl.C