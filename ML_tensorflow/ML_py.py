import uproot
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras.layers import Input, Activation, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import utils
from tensorflow import random as tf_random
from tensorflow.keras.utils import plot_model
import random as python_random

treename = 'tree'
filename = {}
upfile = {}
params = {}

df = {} #dataframe 

# Define ROOT files
filename['sig'] = 'atlas-higgs-challenge-2014-v2-sig.root'
filename['bkg'] = 'atlas-higgs-challenge-2014-v2-bkg.root'

# Adding variables to the dataframe 
VARS = ['DER_mass_MMC', 'DER_mass_transverse_met_lep' , 'DER_mass_vis', 'DER_pt_ratio_lep_tau', 'PRI_tau_pt', 'PRI_met']

NDIM = len(VARS) #check

print("Number of kinematic variables imported from the ROOT files = %d"% NDIM)

upfile['sig'] = uproot.open(filename['sig'])
upfile['bkg'] = uproot.open(filename['bkg'])

# Look at the signal and bkg events 
df['sig'] = pd.DataFrame(upfile['sig'][treename].arrays(VARS, library="np"),columns=VARS)
print(df['sig'].shape)

df['sig'].head()

df['bkg'] = pd.DataFrame(upfile['bkg'][treename].arrays(VARS, library="np"),columns=VARS)
df['bkg'].head()

print(df['bkg'].shape)

# Remove undefined variable entries VARS[i] <= -999
for i in range(NDIM):
    df['sig']= df['sig'][(df['sig'][VARS[i]] > -999)]
    df['bkg']= df['bkg'][(df['bkg'][VARS[i]] > -999)]

# Add the columnisSignal to the dataframe containing the truth information
# i.e. it tells if that particular event is signal (isSignal=1) or background (isSignal=0)
df['sig']['isSignal'] = np.ones(len(df['sig'])) 
df['bkg']['isSignal'] = np.zeros(len(df['bkg'])) 
print("Number of Signal events = %d " %len(df['sig']['isSignal']))
print("Number of Background events = %d " %len(df['bkg']['isSignal']))

#Showing that the variable isSignal is correctly assigned for signal events
print(df['sig']['isSignal'])
print(df['bkg']['isSignal'])

df_all = pd.concat([df['sig'],df['bkg']])

df_all = shuffle(df_all)

#--- Preparing variables for the ML algorithm -------------------------------------------

NN_VARS= ['DER_mass_MMC', 'DER_mass_transverse_met_lep' , 'DER_mass_vis', 'DER_pt_ratio_lep_tau', 'PRI_tau_pt', 'PRI_met']
 
df_input  = df_all.filter(NN_VARS)
df_target = df_all.filter(['isSignal']) 


# Transform dataframes to numpy arrays of float32 
# (X->NN input , Y->NN target output , W-> event weights)

NINPUT=len(NN_VARS)
print("Number NN input variables=",NINPUT)
print("NN input variables=",NN_VARS)
X  = np.asarray( df_input.values ).astype(np.float32)
Y  = np.asarray( df_target.values ).astype(np.float32)
print(X.shape)
print(Y.shape)
print('\n')

#---- Dividing testing and training data ------------------------------------------------
# use of a sklearn algorithm

X_train_val, X_test, Y_train_val , Y_test = train_test_split(X, Y, test_size=0.2,shuffle=True)

'''size= int(len(X[:,0]))
test_size = int(0.2*len(X[:,0]))
print('X (features) before splitting')
print('\n')
print(X.shape)
print('X (features) splitting between test and training')
X_test= X[0:test_size+1,:]
print('Test:')
print(X_test.shape)
X_train_val= X[test_size+1:len(X[:,0]),:]
print('Training:')
print(X_train_val.shape)
print('\n')
print('Y (target) before splitting')
print('\n')
print(Y.shape)
print('Y (target) splitting between test and training ')
Y_test= Y[0:test_size+1,:]
print('Test:')
print(Y_test.shape)
Y_train_val= Y[test_size+1:len(Y[:,0]),:]
print('Training:')
print(Y_train_val.shape)
print('\n')'''


#---- ANN model implementation ----------------------------------------------

# Define Neural Network with 3 hidden layers
input  = Input(shape=(NINPUT,), name = 'input') 
hidden = Dense(NINPUT*10, name = 'hidden1', kernel_initializer='normal', activation='selu')(input)
hidden = Dropout(rate=0.1)(hidden)
hidden = Dense(NINPUT*2 , name = 'hidden2', kernel_initializer='normal', activation='selu')(hidden)
hidden = Dropout(rate=0.1)(hidden)
hidden = Dense(NINPUT, name = 'hidden3', kernel_initializer='normal', activation='selu')(hidden)
hidden = Dropout(rate=0.1)(hidden)
output  = Dense(1 , name = 'output', kernel_initializer='normal', activation='sigmoid')(hidden)


# create the model
model = Model(inputs=input, outputs=output)

optim = RMSprop(lr = 1e-4)

# print learning rate each epoch to see if reduce_LR is working as expected

#def get_lr_metric(optim):
#    def lr(y_true, y_pred):
#        return optim.lr
#    return lr

# compile the model
#model.compile(optimizer=optim, loss='mean_squared_error', metrics=['accuracy'], weighted_metrics=['accuracy'])
#model.compile(optimizer=optim, loss='mean_squared_error', metrics=['accuracy'])
model.compile( optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'], weighted_metrics=['accuracy'])
#accuracy (defined as the number of good matches between the predictions and the class labels)
# print the model summary
model.summary()

plot_model(model, show_shapes=True, show_layer_names=True)

# Save the model
model_file = 'ANN.h5'

# monitor the chosen metrics
checkpoint = keras.callbacks.ModelCheckpoint(filepath = model_file,
                                             monitor = 'val_loss',
                                             mode='min',
                                             save_best_only = True)

#Stop training when a monitored metric has stopped improving
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                           mode='min',# quantity that has to be monitored(to be minimized in this case)
                              patience = 50, # Number of epochs with no improvement after which training will be stopped.
                              min_delta = 1e-7,
                              restore_best_weights = True) # update the model with the best-seen weights

#Reduce learning rate when a metric has stopped improving
reduce_LR = keras.callbacks.ReduceLROnPlateau( monitor = 'val_loss',
                                              mode='min',# quantity that has to be monitored
                                              min_delta=1e-7,
                                              factor = 0.1, # factor by which LR has to be reduced...
                                              patience = 10, #...after waiting this number of epochs with no improvements 
                                              #on monitored quantity
                                              min_lr= 0.00001 ) 


callback_list = [reduce_LR, early_stop, checkpoint]

# Number of training epochs (you can change this number)
nepochs=200 #it will take about 6 minutes

# Batch size
batch=250

history = model.fit(X_train_val[:,0:NINPUT], 
                    Y_train_val,
                    epochs=nepochs, 
                    batch_size=batch,
                    callbacks = callback_list, 
                    verbose=1, # switch to 1 for more verbosity 
                    validation_split=0.3 ) # fix the validation dataset size


model = keras.models.load_model('ANN.h5')


#---- Performance -------------------------------------------

from sklearn.metrics import precision_recall_curve , roc_curve, auc 

# plot the loss fuction vs epoch during the training phase
# the plot of the loss function on the validation set is also computed and plotted
#plt.rcParams['figure.figsize'] = (13,6)
plt.plot(history.history['loss'], label='loss train',color='b')
plt.plot(history.history['val_loss'], label='loss validation',color='r')
plt.title("Loss") #,fontsize=12,fontweight='bold', color='b'
plt.legend(loc="upper right")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# Plot accuracy metrics vs epoch during the training
# for the proper training dataset and the validation one
plt.plot(history.history['accuracy'], label='accuracy train',color='b')
plt.plot(history.history['val_accuracy'], label='accuracy validation',color='r')
plt.title("Accuracy") 
plt.ylim([0, 1.0])
plt.legend(loc="lower left")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('accuracy.png')

# Get ANN model label predictions and performance metrics curves, after having trained the model
y_true = Y_test[:,0]
y_true_train = Y_train_val[:,0]

Y_prediction = model.predict(X_test[:,0:NINPUT])

# Get precision, recall, 
p, r, t = precision_recall_curve( y_true= Y_test, probas_pred= Y_prediction 
                                 #, sample_weight=w_test 
                                 )
# Get False Positive Rate (FPR) True Positive Rate (TPR) , Thresholds/Cut on the ANN's score
fpr, tpr, thresholds = roc_curve( y_true= Y_test,  y_score= Y_prediction 
                                #,sample_weight=w_test 
                                )


Y_prediction_train = model.predict(X_train_val[:,0:NINPUT])
p_train, r_train, t_train = precision_recall_curve( Y_train_val, Y_prediction_train 
                                                   #, sample_weight=w_train 
                                                   )
fpr_train, tpr_train, thresholds_train = roc_curve(Y_train_val, Y_prediction_train
                                                   #, sample_weight=w_train
                                                   )

# Plotting the ANN ROC curve on the test and training datasets
#Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)
roc_auc_train = auc(fpr_train,tpr_train)
plt.plot(fpr_train, tpr_train,  color='b', label='NN AUC_train = %.4f' % (roc_auc_train))
plt.plot(fpr, tpr, color='r', label='NN AUC_test = %.4f' % (roc_auc))

# Comparison with the random chance curve
plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='random chance')
plt.xlim([0, 1.0]) #fpr
plt.ylim([0, 1.0]) #tpr
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)') 
plt.legend(loc="lower right")
plt.show()
plt.savefig('roc.png')

# Plot of the metrics Efficiency x Purity -- ANN 
# Looking at this curve we will choose a threshold on the ANN score
# for distinguishing between signal and background events
plt.plot(t,p[:-1]*r[:-1],label='purity*efficiency_test')
plt.plot(t_train,p_train[:-1]*r_train[:-1],label='purity*efficiency_train')
plt.xlabel('Threshold/cut on the ANN score')
plt.ylabel('Purity*efficiency')
plt.title('Purity*efficiency vs Threshold on the ANN score')
plt.tick_params(width=2, grid_alpha=0.5)
plt.legend(loc="lower left")
plt.show()
plt.savefig('purity*efficiency_test.png')
