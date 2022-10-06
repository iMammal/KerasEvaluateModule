from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
import keras_tuner as kt
import time
import os
from math import sqrt

#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
policy = keras.mixed_precision.Policy('mixed_float16')
#keras.mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

class SequentialModel(kt.HyperModel):
    cvscores = []
    modelscores = []
    predictions = []
    history_dict = {}
    model = keras.Model
    name = "Seq"
    lineformat = '-'
    sp_history = []
    sn_history = []
    auc_history = []
    acc_history = []
    mcc_history = []
    Fscore_history = []
    pre_history = []

    confusion_history = []

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.name = "Seq"
        self.lineformat = '-'

    def build(self, hp):
        """Builds a Sequential model."""

        inputs = keras.Input(self.input_shape)    #shape=(28, 28, 1))
        x = keras.layers.Flatten()(inputs)
        ###hpunits=hp.Choice("units", [40, 140, 90])   ### 32, 64, 128
        hpunits=[40, 140, 90]   ### 32, 64, 128
        x = keras.layers.Dense(units=hpunits[0], activation="relu")(x)
        x = keras.layers.Dense(units=hpunits[1], activation="relu")(x)
        x = keras.layers.Dense(units=hpunits[2], activation="relu")(x)
        print("Added keras layers with units",hpunits)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)

        # model.add(Dense(1, activation='sigmoid'))

        self.model = keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self.model

        # model.compile(loss, optimizer, metrics)

    def modelevaluate(self, xtest, ytest, verbose=0):

        self.modelscores = self.model.evaluate(xtest, ytest, verbose=0)

        return self.modelscores

    def modelcvscores(self):
        print("%s: %.2f%%" % (self.model.metrics_names[1], self.modelscores[1] * 100))
        self.cvscores.append(self.modelscores[1] * 100)

        self.history_dict[self.model.name] = [self.history_callback, self.model]
        # model = history_dict[model_name][1]

    def modelpredict(self, xtest, verbose=0):
        self.predictions = self.model.predict(xtest)
        return self.predictions

    def printstats(self,fpr,tpr, tn, fp, fn, tp, sp, sn, pre, mcc, acc, Fscore):
        print("Seq: %.2f%%",auc(fpr,tpr))
        print("Seq: %.2f%% (+/- %.2f%%)" % (np.mean(self.cvscores), np.std(self.cvscores)))
        print("Seq: FP %.2f%%	FN %.2f%%	TP %.2f%%	TN %.2f%%",fp,fn,tp,tn)
        print("Seq: SP %.2f%%	SN %.2f%%	Pre %.2f%%	MCC %.2f%%	Acc %.2f%%	Fscore %.2f%%",sp,sn,pre,mcc,acc,Fscore)
        self.auc_history.append(auc(fpr,tpr))
        self.confusion_history.append([fp,fn,tp,tn])
        self.acc_history.append(acc)
        self.mcc_history.append(mcc)
        self.Fscore_history.append(Fscore)
        self.pre_history.append(pre)
        self.sp_history.append(sp)
        self.sn_history.append(sn)

    # def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):
    #    super(kt.HyperModel,self).fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs)

    def fitmodel(self,X,Y,vd,cb):
        self.history_callback = self.model.fit(X, Y,
                  epochs=50,
                  batch_size=10,
                  verbose=0,
                  validation_data=vd,
                  callbacks=[cb])

        return self.history_callback

class RandomForestRegressorModel(kt.HyperModel):
    cvscores = []
    modelscores = []
    predictions = []
    history_dict = {}
    model = keras.Model
    name = "RF"
    lineformat = '--'
    sp_history = []
    sn_history = []
    auc_history = []
    acc_history = []
    mcc_history = []
    Fscore_history = []
    pre_history = []
    confusion_history = []

    def __init__(self):
        self.name = "RF"
        self.lineformat = '--'

    def build(self, hp):
        self.model = RandomForestRegressor(n_estimators=20, random_state=0)
        return self.model
    def fitmodel(self,X,Y,vd=[],cb=[]):
        self.model.fit(X, Y)

    def modelcvscores(self):
        # print("%s: %.2f%%" % (self.model.metrics_names[1], self.modelscores[1] * 100))
        # self.cvscores.append(self.modelscores[1] * 100)

        # self.history_dict[model.name] = [self.history_callback, self.model]
        # model = history_dict[model_name][1]
        return

    def modelpredict(self, xtest, verbose=0):
        self.predictions = self.model.predict(xtest)
        return self.predictions
    def modelevaluate(self, xtest, ytest, verbose=0):
        # self.modelscores = model.score(xtest, ytest)
        return self.modelscores

    def printstats(self,fpr,tpr, tn, fp, fn, tp, sp, sn, pre, mcc, acc, Fscore):
        # print("%.2f%% (+/- %.2f%%)" % (np.mean(self.cvscores), np.std(self.cvscores)))
        #print("RF stats...")
        print("RF: AUC %.2f%%",auc(fpr,tpr))
        print("RF: FP %.2f%%	FN %.2f%%	TP %.2f%%	TN %.2f%%",fp,fn,tp,tn)
        print("RF: SP %.2f%%	SN %.2f%%	Pre %.2f%%	MCC %.2f%%	Acc %.2f%%	Fscore %.2f%%",sp,sn,pre,mcc,acc,Fscore)
        self.auc_history.append(auc(fpr,tpr))
        self.confusion_history.append([fp,fn,tp,tn])
        self.acc_history.append(acc)
        self.mcc_history.append(mcc)
        self.Fscore_history.append(Fscore)
        self.pre_history.append(pre)
        self.sp_history.append(sp)
        self.sn_history.append(sn)

class SVMModel(kt.HyperModel):
    cvscores = []
    modelscores = []
    predictions = []
    history_dict = {}
    model = keras.Model
    name = "SVM"
    lineformat = ':'
    # model = SVC(kernel='linear')
    sp_history = []
    sn_history = []
    auc_history = []
    acc_history = []
    mcc_history = []
    Fscore_history = []
    pre_history = []
    confusion_history = []

    def __init__(self):
        self.name = "SVM"
        self.lineformat = ':'


    def build(self, hp):
        self.model = SVC(kernel='linear')
        return self.model
    def fitmodel(self,X,Y,vd=[],cb=[]):
        self.model.fit(X, Y)

    def modelcvscores(self):
        # print("%s: %.2f%%" % (self.model.metrics_names[1], self.modelscores[1] * 100))
        # self.cvscores.append(self.modelscores[1] * 100)

        # self.history_dict[model.name] = [self.history_callback, self.model]
        # model = history_dict[model_name][1]
        return

    def modelpredict(self, xtest, verbose=0):
        self.predictions = self.model.predict(xtest)
        return self.predictions
    def modelevaluate(self, xtest, ytest, verbose=0):
        # self.modelscores = model.score(xtest, ytest)
        return self.modelscores

    def printstats(self,fpr,tpr, tn, fp, fn, sp,sn,tp,pre,acc,mcc,Fscore):
        # print("%.2f%% (+/- %.2f%%)" % (np.mean(self.cvscores), np.std(self.cvscores)))
        print("SVM: %.2f%%",auc(fpr,tpr))
        print("SVM: FP %.2f%%	FN %.2f%%	TP %.2f%%	TN %.2f%%",fp,fn,tp,tn)
        print("SVM: SP %.2f%%	SN %.2f%%	Pre %.2f%%	Acc %.2f%%	MCC %.2f%%	Fscore %.2f%%",sp,sn,pre,acc,mcc,Fscore)
        #print("SVM stats...")
        self.auc_history.append(auc(fpr,tpr))
        self.confusion_history.append([fp,fn,tp,tn])
        self.acc_history.append(acc)
        self.mcc_history.append(mcc)
        self.Fscore_history.append(Fscore)
        self.pre_history.append(pre)
        self.sp_history.append(sp)
        self.sn_history.append(sn)

def build_sequential_model():
    # global model
    model = Sequential()
    model.add(Dense(16, input_dim=5, activation='relu'))
    # model.add(Dense(20, activation='relu'))
    for i in range(modelDepth):
        model.add(Dense(80 - 20 * i, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # model.name = "Dense2Layers"

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def roundnum(x):
    if (x>0.5):
      return 1
    else:
      return 0


plt.style.use('ggplot')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# TensorBoard Callback
cb = TensorBoard()

# load Gencode  dataset
dataset = np.loadtxt("NewGencode4DLTraining.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:, 0:5]
Y = dataset[:, 5]

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# k = 0

hypermodel = {}

for train, test in kfold.split(X, Y):

    # modelDepth = k % 3
    # k += 1

    # create model
    inithp = kt.HyperParameters()

    hypermodel[0] = SVMModel()
    model = hypermodel[0].build(inithp)

    input_shape = (X[train].shape[1],)

    hypermodel[1] = SequentialModel(input_shape)
    model = hypermodel[1].build(inithp)

    hypermodel[2] = RandomForestRegressorModel()
    model = hypermodel[2].build(inithp)

    # Fit the model
    history_callback = hypermodel[1].fitmodel(X[train], Y[train],(X[test], Y[test]),cb)
    hypermodel[0].fitmodel(X[train], Y[train])
    hypermodel[2].fitmodel(X[train], Y[train])

    # evaluate the model

    for k in range(0,3):
        hypermodel[k].modelevaluate(X[test], Y[test], verbose=0)

        hypermodel[k].modelcvscores()

        Y_pred = hypermodel[k].modelpredict(X[test])
        fpr, tpr, threshold = roc_curve(Y[test].ravel(), Y_pred.ravel())

        #print(Y[test].ravel())
        #print(Y_pred.ravel())
        Y_predbin = [roundnum(y) for y in Y_pred.ravel()]
        #print(Y_predbin)

        tn, fp, fn, tp = confusion_matrix(Y[test].ravel(), Y_predbin).ravel()
        sp = tn / (tn + fp)
        sn = tp / (tp + fn)
        pre = tp / (tp + fp)
        ACC = (tp + tn) / (tp + tn + fp + fn)
        Fscore = (2 * pre * sn) / (pre + sn)
        MCC = (tp * tn - fp * tn) / sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))



        plt.plot(fpr, tpr, hypermodel[k].lineformat, label='{}, AUC = {:.3f}'.format(hypermodel[k].name, auc(fpr, tpr)))

        hypermodel[k].printstats(fpr, tpr, tn, fp, fn, tp,sp,sn,pre,ACC,Fscore,MCC)

    # else:
    #    print("SVM: %.2f",auc(fpr,tpr))
    #    plt.plot(fpr, tpr, label='{}, AUC = {:.3f}'.format("SVM", auc(fpr, tpr)))

# plt.figure(figsize=(10, 10))
# plt.plot([0, 1], [0, 1], 'k--')

# for model_name in history_dict:

csv_results_folder = "csv-results"
date_now = time.strftime("%Y-%m-%d")

auc_history = [hypermodel[0].auc_history, hypermodel[1].auc_history, hypermodel[2].auc_history]
pre_history = [hypermodel[0].pre_history, hypermodel[1].pre_history, hypermodel[2].pre_history]
acc_history = [hypermodel[0].acc_history, hypermodel[1].acc_history, hypermodel[2].acc_history]
mcc_history = [hypermodel[0].mcc_history, hypermodel[1].mcc_history, hypermodel[2].mcc_history]
Fscore_history = [hypermodel[0].Fscore_history, hypermodel[1].Fscore_history, hypermodel[2].Fscore_history]
sp_history = [hypermodel[0].sp_history, hypermodel[1].sp_history, hypermodel[2].sp_history]
sn_history = [hypermodel[0].sn_history, hypermodel[1].sn_history, hypermodel[2].sn_history]

auc_means = [np.mean(hypermodel[0].auc_history), np.mean(hypermodel[1].auc_history), np.mean(hypermodel[2].auc_history)] #, "AUC"]
sp_means = [np.mean(hypermodel[0].sp_history), np.mean(hypermodel[1].sp_history), np.mean(hypermodel[2].sp_history)] #, "Specificity"]
sn_means = [np.mean(hypermodel[0].sn_history), np.mean(hypermodel[1].sn_history), np.mean(hypermodel[2].sn_history)] #, "Sensitivity"]
pre_means = [np.mean(hypermodel[0].pre_history), np.mean(hypermodel[1].pre_history), np.mean(hypermodel[2].pre_history)] #, "Precision"]
acc_means = [np.mean(hypermodel[0].acc_history), np.mean(hypermodel[1].acc_history), np.mean(hypermodel[2].acc_history)] #, "Accuracy"]
mcc_means = [np.mean(hypermodel[0].mcc_history), np.mean(hypermodel[1].mcc_history), np.mean(hypermodel[2].mcc_history)] #, "MCC"]
Fscore_means = [np.mean(hypermodel[0].Fscore_history), np.mean(hypermodel[1].Fscore_history), np.mean(hypermodel[2].Fscore_history)] #, "Fscore"]

#means_array = ([auc_means, sp_means, sn_means, pre_means, acc_means, mcc_means, Fscore_means])
means_array = np.array([auc_means, sp_means, sn_means, pre_means, acc_means, mcc_means, Fscore_means])

confusion_history = [hypermodel[0].confusion_history,hypermodel[1].confusion_history,hypermodel[2].confusion_history]

auc_history = np.array(auc_history)
confusion_history = np.array(confusion_history)
pre_history = np.array(pre_history)
acc_history = np.array(acc_history)
mcc_history = np.array(mcc_history)
Fscore_history = np.array(Fscore_history)
sp_history = np.array(sp_history)
sn_history = np.array(sn_history)

#for k in range(0,3):
#auc_history = np.append([auc_history,hypermodel[0].auc_history, auc_history,hypermodel[1].auc_history, auc_history,hypermodel[2].auc_history], axis=0)
#confusion_history=np.append([confusion_history,hypermodel[0].confusion_history,confusion_history,hypermodel[1].confusion_history,confusion_history,hypermodel[2].confusion_history], axis=0)

if os.environ.get('OS','') == "Windows_NT":
    #auc_history.tofile(f"{csv_results_folder}\evaluate_auc_{date_now}.csv", ",")
    #confusion_history.tofile(f"{csv_results_folder}\evaluate_confusion_{date_now}.csv", ",")
    np.savetxt(f"{csv_results_folder}\evaluate_auc_{date_now}.csv", auc_history, delimiter=",",fmt='%-7.3f')
    #np.savetxt(f"{csv_results_folder}\evaluate_confusion_{date_now}.csv", confusion_history, delimiter=",",fmt='%-7.3f')
    np.savetxt(f"{csv_results_folder}\evaluate_pre_{date_now}.csv", pre_history, delimiter=",",fmt='%-7.3f')
    np.savetxt(f"{csv_results_folder}\evaluate_acc_{date_now}.csv", acc_history, delimiter=",",fmt='%-7.3f')
    np.savetxt(f"{csv_results_folder}\evaluate_mcc_{date_now}.csv", mcc_history, delimiter=",",fmt='%-7.3f')
    np.savetxt(f"{csv_results_folder}\evaluate_Fscore_{date_now}.csv", Fscore_history, delimiter=",",fmt='%-7.3f')
    np.savetxt(f"{csv_results_folder}\evaluate_meanstats_{date_now}.csv", means_array, delimiter=",", fmt='%-7.3f', header="SVM,DNN,RFR", footer="Rows: AUC,SP,SN,PRE,ACC,MCC,FSCORE")
    np.savetxt(f"{csv_results_folder}\evaluate_sp_{date_now}.csv", sp_history, delimiter=",",fmt='%-7.3f')
    np.savetxt(f"{csv_results_folder}\evaluate_sn_{date_now}.csv", sn_history, delimiter=",",fmt='%-7.3f')
else:
    np.savetxt(f"{csv_results_folder}/evaluate_auc_{date_now}.csv", auc_history, delimiter=",",fmt='%-7.3f')
    #np.savetxt(f"{csv_results_folder}/evaluate_confusion_{date_now}.csv", confusion_history, delimiter=",",fmt='%-7.3f')
    np.savetxt(f"{csv_results_folder}/evaluate_pre_{date_now}.csv", pre_history, delimiter=",",fmt='%-7.3f')
    np.savetxt(f"{csv_results_folder}/evaluate_acc_{date_now}.csv", acc_history, delimiter=",",fmt='%-7.3f')
    np.savetxt(f"{csv_results_folder}/evaluate_mcc_{date_now}.csv", mcc_history, delimiter=",",fmt='%-7.3f')
    np.savetxt(f"{csv_results_folder}/evaluate_Fscore_{date_now}.csv", Fscore_history, delimiter=",",fmt='%-7.3f')
    np.savetxt(f"{csv_results_folder}/evaluate_meanstats_{date_now}.csv", means_array, delimiter=",", fmt='%-7.3f', header="SVM,DNN,RFR", footer="Rows: AUC,SP,SN,PRE,ACC,MCC,FSCORE")
    np.savetxt(f"{csv_results_folder}/evaluate_sp_{date_now}.csv", sp_history, delimiter=",",fmt='%-7.3f')
    np.savetxt(f"{csv_results_folder}/evaluate_sn_{date_now}.csv", sn_history, delimiter=",",fmt='%-7.3f')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend()
plt.show()
