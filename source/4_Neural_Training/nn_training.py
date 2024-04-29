# -*- coding: utf-8 -*-
#LIBRARY IMPORTS


import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)



#Needed for normal data manipulation
from extract_data import extract,unshuffle
import numpy as np
import math

#Necessary sci kit learn packages for Machine learning programming
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor

#Metrics for quality verification and data normalilsation
import sklearn.metrics as met
from sklearn.model_selection import train_test_split,KFold
from statistics import mean, stdev

#convenient progress bar 
from tqdm import tqdm


#%%
#NUMPY DATA

X1 = np.transpose(extract(os.path.join(current_dir, "X1.txt")))
Y1 = np.transpose(extract(os.path.join(current_dir, "Y1.txt")))

X2 = np.transpose(extract(os.path.join(current_dir, "X2.txt")))
Y2 = np.transpose(extract(os.path.join(current_dir, "Y2.txt")))

X = np.concatenate((X1,X2),axis = 0)
Y = np.concatenate((Y1,Y2),axis = 0)

Y_x = Y[:,0]
Y_z = Y[:,2]

del Y

#%%
#NUMPY --> PANDAS FRAME

Y_xz = pd.DataFrame({'x direction' : Y_x[:],'z direction':Y_z[:]})

X = pd.DataFrame({'electrode 1':X[:,0],'electrode 2':X[:,1],
                   'electrode 3':X[:,2],'electrode 4':X[:,3],
                   'electrode 5':X[:,4],'electrode 6':X[:,5],
                   'electrode 7':X[:,6],'electrode 8':X[:,7],
                   'electrode 9':X[:,8],'electrode 10':X[:,9],
                   'electrode 11':X[:,10],'electrode 12':X[:,11],
                   'electrode 13':X[:,12],'electrode 14':X[:,13],
                   'electrode 15':X[:,14],'electrode 16':X[:,15],
                   'electrode 17':X[:,16],'electrode 18':X[:,17],
                   'electrode 19':X[:,18]})
del Y_x
del Y_z

#%%
#SPLITTING & SCALING

#We will use the x and z directions for learning (y is deducable from x and z)
indices = np.array(range(len(Y_xz)))
X_train, X_validation, Y_xz_train, Y_xz_validation,indices_train,indices_validation = train_test_split(X,Y_xz,indices,
                                              test_size = 0.5,
                                              random_state = 42,
                                              shuffle = False)
Y_xz_train = Y_xz_train.to_numpy()
Y_xz_validation = Y_xz_validation.to_numpy()

X_train = X_train.to_numpy()
X_validation = X_validation.to_numpy()

X = X.to_numpy()
Y_xz = Y_xz.to_numpy()



#%%
# Grid Searching for best model parameters(feel free to add/subtract more
# but keep in mind it takes more time to search. The current defaults are the
# optimal ones)
NL = [7]
NPL = [5]
LR = [0.01, 0.011,0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02]
MI = [500]
ALPH = [0.0001]

#This doesnt change, it works best no matter what
LT = 'adaptive'

nkfold = 5
kf = KFold(n_splits = 5,shuffle=False)
kf.get_n_splits(X_train)
metrics_file = pd.DataFrame(columns = ['#_Layers','#_Nodes','Learning_Rate',
                                       'Max_reuse', 'L2_Penalty',
                                       'R^2','sigma_R^2',
                                       'AIC', 'sigma_AIC',
                                       'Explained_Variance', 'sigma_Explained_Variance',
                                       'Mean_Squared_Error', 'sigma_Mean_Squared_Error'])

total_perm = len(NL)*len(NPL)*len(LR)*len(MI)*len(ALPH)*nkfold
pbar = tqdm(total = total_perm, desc="Grid Coverage", ncols = 100, leave=False)

for Layers in NL:
    for Nodes in NPL:
        for Learn in LR:
            for Maxuse in MI:
                for L2Penalty in ALPH: 
                    R2 = []
                    AIC = []
                    EV = []
                    MAXE = []
                    MAE = []
                    MSE = []
                    MEDAE = []
                    RNN1 = MLPRegressor(hidden_layer_sizes=(Nodes,)*Layers,
                                        max_iter=Maxuse,
                                        learning_rate_init=Learn,
                                        alpha = L2Penalty,
                                        learning_rate = LT)
                   # print('Now testing Layers = {},\nNodes = {},\nLearn = {},\nMaxuse = {},\nL2Penalty = {}'.format(Layers,Nodes,Learn,Maxuse,L2Penalty))
                    for train_index,test_index in kf.split(X_train):
                        
                        x_train = X_train[train_index,:]
                        x_test =  X_train[test_index,:]
                        y_train = Y_xz_train[train_index,:]
                        y_test =  Y_xz_train[test_index,:]
                        
                        RNN1.fit(X_train,Y_xz_train)
                        
                        y_pred = RNN1.predict(x_test)
                        
                        r2 = met.r2_score(y_test,y_pred)
                        #print(r2)
                        R2.append(r2)
                        
                        exvar = met.explained_variance_score(y_test,y_pred)
                        EV.append(exvar)
                        
                        meansqerr = met.mean_squared_error(y_test,y_pred)
                        MSE.append(meansqerr)
                        
                        AIC2 =  2*(Layers*(Nodes**2+Nodes) - math.log(1-r2)) 
                        AIC.append(AIC2)
                        pbar.update(1)
                    #print('{}% Done'.format(round(running_total/total_perm*100,2)))
                    #print(' ')
                    ##%%
                    #METRIC EVALUATION
                    metrics_row =           {'#_Layers':Layers,
                                         '#_Nodes':Nodes,
                                         'Learning_Rate':Learn,
                                         'Max_reuse':Maxuse,
                                         'L2_Penalty':L2Penalty,
                                         'R^2':max(R2),
                                         'sigma_R^2':stdev(R2),
                                         'AIC':max(AIC),
                                         'sigma_AIC':stdev(AIC),
                                         'Explained_Variance':mean(EV),
                                         'sigma_Explained_Variance':stdev(EV),
                                         'Mean_Squared_Error':mean(MSE),
                                         'sigma_Mean_Squared_Error':stdev(MSE)}
                    metrics_file.loc[len(metrics_file)] = metrics_row
pbar.close()
#%%FINAL VALIDATION
metrics_file.to_csv(os.path.join(current_dir, "hyperparameter_grid_search_overview.csv"))
idx_best = metrics_file['AIC'].idxmax()# change to idxmin when the number of layers/neurons per layer varies in the grid

row_best = metrics_file[idx_best:(idx_best+1)]
#%%

#Current Optimum: (feel free to manualy tune or to explore the csv for alternatives)
Nodes = row_best["#_Nodes"].to_numpy()[0]
Layers = row_best["#_Layers"].to_numpy()[0]
Learn = row_best["Learning_Rate"].to_numpy()[0]
Iterations = row_best["Max_reuse"].to_numpy()[0]
alph = row_best["L2_Penalty"].to_numpy()[0]
LT = 'adaptive'

optcoeff =  []
optint = []
optr2 = [0,0]
opttrain_loss = []
opttest_loss = []
opttrain_r2 = []
opttest_r2 = []

myr = 10
n_epochs = 50
pbar = tqdm(total = myr*n_epochs, desc="Weight Finder", ncols = 100, leave=False)
for i in range(myr):
    RNN1 = MLPRegressor(hidden_layer_sizes=(Nodes,)*Layers,
                        max_iter=Iterations,
                        learning_rate_init=0.01,
                        alpha = 0.0001,
                        learning_rate = LT)
    
    X_train_refine, X_test, Y_xz_train_refine, Y_xz_test = train_test_split(X_train,Y_xz_train, train_size = 0.5)
    
    train_loss = []
    test_loss = []
    train_r2 = []
    test_r2 = []
    
    for i in range(n_epochs):
        
        RNN1.partial_fit(X_train_refine, Y_xz_train_refine)
        
        train_loss.append(met.mean_squared_error(Y_xz_train_refine, RNN1.predict(X_train_refine)))
        test_loss.append(met.mean_squared_error(Y_xz_test, RNN1.predict(X_test)))
        
        train_r2.append(met.r2_score(Y_xz_train_refine, RNN1.predict(X_train_refine)))
        test_r2.append(met.r2_score(Y_xz_test, RNN1.predict(X_test)))
        
        pbar.update(1)
        
    #RNN1.fit(X_train_refine,Y_xz_train_refine)
    Y_xz_pred = RNN1.predict(X_test)
    #Y_xz_pred_all = RNN1.predict(X)
    
    
    r2 = [met.r2_score(Y_xz_test[:,0],Y_xz_pred[:,0]),
          met.r2_score(Y_xz_test[:,1],Y_xz_pred[:,1])]
    
    if((r2[0] + r2[1])/2 > (optr2[0] +  optr2[1])/2):
        
        optcoeff = RNN1.coefs_
        optint = RNN1.intercepts_
        
        opttrain_loss = train_loss
        opttest_loss = test_loss
        opttrain_r2 = train_r2
        opttest_r2 = test_r2
        
        optr2 = r2
        
          

pbar.close()
print('')
print(optr2)

 #%%
RNN1.coefs_ = optcoeff
RNN1.intercepts_ = optint
Y_xz_pred = RNN1.predict(X_validation)
r2_validation = [met.r2_score(Y_xz_validation[:,0],Y_xz_pred[:,0]),
                 met.r2_score(Y_xz_validation[:,1],Y_xz_pred[:,1])]


print('')
print(r2_validation)

Y_xz_pred = RNN1.predict(X_validation)
Y_xz_pred_all = RNN1.predict(X)


#%%
# Plot the loss and R^2 curves
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, n_epochs + 1), opttrain_loss, label="Training Loss", color="blue")
plt.plot(range(1, n_epochs+ 1), opttest_loss, label="Test Loss", color="red")
plt.title("Training and Validation Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss (Mean Squared Error)")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, n_epochs + 1), opttrain_r2, label="Training $R^2$", color="blue")
plt.plot(range(1, n_epochs+ 1), opttest_r2, label="Test $R^2$", color="red")
plt.plot(n_epochs, np.mean(r2_validation), 'gx', label = "Final Validation $R^2$")
plt.title("Training and Validation $R^2$ Curves")
plt.xlabel("Epochs")
plt.ylabel("$R^2$ Score")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

#%%  Save the Weights
weights_dir = os.path.join(os.path.dirname(current_dir), "5_Final_Weights")

for i,weight in enumerate(optcoeff):
    np.savetxt(os.path.join(weights_dir, "coeff"+str(i)+".csv"), weight, delimiter = ',')
    
for i, bias in enumerate(optint):
    np.savetxt(os.path.join(weights_dir, "bias"+str(i)+".csv"), weight, delimiter = ',')
    
#%%            
#exvar = met.explained_variance_score(Y_xz_validation,Y_xz_pred)
#maxerr = met.max_error(Y_x_ztest,Y_xz_pred)
#meanabserr = met.mean_absolute_error(Y_xz_validation,Y_xz_pred)
#meansqerr = met.mean_squared_error(Y_xz_validation,Y_xz_pred)
#medianabserr = met.median_absolute_error(Y_xz_validation,Y_xz_pred)
##AIC2 = -100*math.log(abs(r2)) + nsuppvec/50*19#-2*math.log10(r2)+nsuppvec*19
#
#validation_file = validation_file.append({'Gamma':Optimal_Gamma,'Epsilon':Optimal_Epsilon,'Penalty':Optimal_Penalty,
#                     'R^2':r2,
#                     'AIC2':AIC2,
#                     'Explained_Variance':exvar,
#                     'Maximum_Error':maxerr,
#                     'Mean_Absolute_Error':meanabserr,
#                     'Mean_Squared_Error':meansqerr,
#                     'Median_Absolute_Error':medianabserr,
#                     '#_Support_Vectors':nsuppvec,
#                     'SupportV/Training_Ratio':suppvecrat},ignore_index = True)

plotting_index = 0
plt.figure(figsize = [16,8])
plt.subplot(221)
plt.title('Z Training data over index(time)')
redd, = plt.plot(unshuffle(indices_train,indices_train),
                 unshuffle(Y_xz_train[:,plotting_index],indices_train),'ro')
plt.xlabel('index number(proportional to time)')
plt.ylabel('Z')
plt.legend([redd],['Z training data'])
plt.tight_layout()


temp_x = [np.arange(-1.0, 0.3, 0.1),np.arange(0.3, 1.2, 0.1)]

plt.subplot(222)
plt.title('Phase plot of Z true against Z predicted')
#plt.title('Gam = {}, Eps = {}, C = {}'.format(Optimal_Gamma,Optimal_Epsilon,Optimal_Penalty))
y_ts_y_p, = plt.plot(Y_xz_validation[:,plotting_index],Y_xz_pred[:,plotting_index],'ro')
yy, = plt.plot(temp_x[plotting_index],temp_x[plotting_index],'b--')
plt.legend([y_ts_y_p,yy],['(Z true,Z pred)','Z true = Z pred'])
plt.xlabel('Z true')
plt.ylabel('Z predicted')
plt.tight_layout()

plt.subplot(223)
plt.title('Z True and Z predicted data over index(time), R^2 = 0.82')
redd, = plt.plot(unshuffle(indices_validation,indices_validation),
                 unshuffle(Y_xz_validation[:,plotting_index],indices_validation),'ro')
blued, = plt.plot(unshuffle(indices_validation,indices_validation),
                  unshuffle(Y_xz_pred[:,plotting_index],indices_validation),'go',alpha = 0.3)
plt.xlabel('index number(proportional to time)')
plt.ylabel('Z')
plt.legend([redd,blued],['Z true','Z predicted'])
plt.tight_layout()

plt.subplot(224)
plt.title('Prediction of the entire dataset')
redd, = plt.plot(Y_xz[:,plotting_index],'r')
blued, = plt.plot(Y_xz_pred_all[:,plotting_index],'g',alpha = 0.7)
plt.xlabel('index number(proportional to time)')
plt.ylabel('Z')
plt.legend([redd,blued],['Z true','Z predicted'])
plt.tight_layout()


#macroplot(X_train,indices_train,suppvec)





