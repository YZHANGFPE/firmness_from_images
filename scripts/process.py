import yaml
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score, r2_score, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt

features = np.load('features.npy')
with open("label.yaml", 'r') as stream:
        labels = yaml.load(stream)

size = len(features)
X = np.ones([size, 4096]) 
Y = np.ones(size)

for i in range(size):
    X[i] = features[i]['feature']
    Y[i] = labels[features[i]['category']]
    
with open("category.txt", 'w') as f:
    for i in range(size):
        f.write(str(i) + '  ' + features[i]['category'] + '\n')

X_te = X[723:]
Y_te = Y[723:]
X_tr = X[:620]
Y_tr = Y[:620]
X_de = X[620 : 723]
Y_de = Y[620 : 723]
cat_te = features[723:]

svr_poly = SVR(kernel='poly', C=0.005, degree=2)
y_poly_te = svr_poly.fit(X_tr, Y_tr).predict(X_te)
y_fix = np.array(Y_te)
y_fix[...] = 5
poly_te_err = mean_absolute_error(Y_te, y_poly_te)
fix_err = mean_absolute_error(Y_te, y_fix)
print "Fix error is: ", fix_err
print "Poly error is: ", poly_te_err
print "Fix median error is: ", median_absolute_error(Y_te, y_fix)
print "Poly median error is: ", median_absolute_error(Y_te, y_poly_te)
print "Explained variance: ", explained_variance_score(Y_te, y_poly_te)

with open("output.txt", 'w') as f:
    for i in range(len(Y_te)):
        f.write(cat_te[i]['category'] + '  ' + str(y_poly_te[i]) + '  ' + str(Y_te[i])  + '\n')

plot = False
if plot:
    plot_x = [1e-6, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    plot_tr_err_rbf = []
    plot_tr_err_lin = []
    plot_tr_err_pol = []
    plot_de_err_rbf = []
    plot_de_err_lin = []
    plot_de_err_pol = []

    for c in plot_x:
        # Fit regression model
        print "penalty C is: ", c
        svr_rbf = SVR(kernel='rbf', C=c, gamma=0.1)
        svr_lin = SVR(kernel='linear', C=c)
        svr_poly = SVR(kernel='poly', C=c, degree=2)
        y_rbf = svr_rbf.fit(X_tr, Y_tr).predict(X_de)
        y_lin = svr_lin.fit(X_tr, Y_tr).predict(X_de)
        y_poly = svr_poly.fit(X_tr, Y_tr).predict(X_de)
        y_rbf_tr = svr_rbf.fit(X_tr, Y_tr).predict(X_tr)
        y_lin_tr = svr_lin.fit(X_tr, Y_tr).predict(X_tr)
        y_poly_tr = svr_poly.fit(X_tr, Y_tr).predict(X_tr)

        # Evaluate the result
        # print "explained_variance_score"
        # print "rbf: ", explained_variance_score(Y_de, y_rbf)
        # print "lin: ", explained_variance_score(Y_de, y_lin)
        # print "poly: ", explained_variance_score(Y_de, y_poly)
        rbf_tr_err = mean_absolute_error(Y_tr, y_rbf_tr)
        lin_tr_err = mean_absolute_error(Y_tr, y_lin_tr)
        poly_tr_err = mean_absolute_error(Y_tr, y_poly_tr)
        rbf_de_err = mean_absolute_error(Y_de, y_rbf)
        lin_de_err = mean_absolute_error(Y_de, y_lin)
        poly_de_err = mean_absolute_error(Y_de, y_poly)
          
        plot_tr_err_rbf.append(rbf_tr_err )
        plot_tr_err_lin.append(lin_tr_err )
        plot_tr_err_pol.append(poly_tr_err)
        plot_de_err_rbf.append(rbf_de_err )
        plot_de_err_lin.append(lin_de_err )
        plot_de_err_pol.append(poly_de_err)
         
        print "training R2 score/mean_absolute_error"
        print "rbf: ", r2_score(Y_tr, y_rbf_tr), rbf_tr_err
        print "lin: ", r2_score(Y_tr, y_lin_tr), lin_tr_err
        print "poly: ", r2_score(Y_tr, y_poly_tr), poly_tr_err 
        print "testing R2 score/mean_absolute_error"
        print "rbf: ", r2_score(Y_de, y_rbf), rbf_de_err
        print "lin: ", r2_score(Y_de, y_lin), lin_de_err
        print "poly: ", r2_score(Y_de, y_poly), poly_de_err

        # with open('output.txt', 'w') as f:
        #     for i in range(200):
        #         f.write(features[i]['category'] + '  ' + str(y_lin[i]) + '  ' + str(Y_de[i]) + '\n')

    # plot the training error and dev error over c
    plt.plot(plot_x, plot_tr_err_rbf)
    plt.plot(plot_x, plot_tr_err_lin)
    plt.plot(plot_x, plot_tr_err_pol)
    plt.plot(plot_x, plot_de_err_rbf, '--')
    plt.plot(plot_x, plot_de_err_lin, '--')
    plt.plot(plot_x, plot_de_err_pol, '--')
    plt.legend(['tr_err_rbf', 'tr_err_lin', 'tr_err_pol', 'de_err_rbf', 'de_err_lin', 'de_err_pol'])
    plt.xlabel('Penalty parameter')
    plt.ylabel('Mean absolute error')
    plt.show()
