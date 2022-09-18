from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
import keras_tuner as kt
import time
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)

class SequentialModel(kt.HyperModel):
    cvscores = []
    modelscores = []
    predictions = []
    history_dict = {}
    model = keras.Model
    name = "Seq"
    lineformat = '-'
    auc_history = []

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.name = "Seq"
        self.lineformat = '-'

    def build(self, hp):
        """Builds a Sequential model."""

        inputs = keras.Input(self.input_shape)    #shape=(28, 28, 1))
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(
            units=hp.Choice("units", [32, 64, 128]), activation="relu"
        )(x)
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

    def printstats(self,fpr,tpr):
        print("Seq: %.2f",auc(fpr,tpr))
        print("%.2f%% (+/- %.2f%%)" % (np.mean(self.cvscores), np.std(self.cvscores)))
        self.auc_history.append(auc(fpr,tpr))

    # def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):
    #    super(kt.HyperModel,self).fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs)

    def fitmodel(self,X,Y,vd,cb):
        self.history_callback = self.model.fit(X, Y,
                  epochs=150,
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
    auc_history = []

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

    def printstats(self,fpr,tpr):
        # print("%.2f%% (+/- %.2f%%)" % (np.mean(self.cvscores), np.std(self.cvscores)))
        #print("RF stats...")
        print("RF: %.2f",auc(fpr,tpr))
        self.auc_history.append(auc(fpr,tpr))

class SVMModel(kt.HyperModel):
    cvscores = []
    modelscores = []
    predictions = []
    history_dict = {}
    model = keras.Model
    name = "SVM"
    lineformat = ':'
    # model = SVC(kernel='linear')
    auc_history = []

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

    def printstats(self,fpr,tpr):
        # print("%.2f%% (+/- %.2f%%)" % (np.mean(self.cvscores), np.std(self.cvscores)))
        print("SVM: %.2f",auc(fpr,tpr))
        #print("SVM stats...")
        self.auc_history.append(auc(fpr,tpr))


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

        plt.plot(fpr, tpr, hypermodel[k].lineformat, label='{}, AUC = {:.3f}'.format(hypermodel[k].name, auc(fpr, tpr)))

        hypermodel[k].printstats(fpr, tpr)

    # else:
    #    print("SVM: %.2f",auc(fpr,tpr))
    #    plt.plot(fpr, tpr, label='{}, AUC = {:.3f}'.format("SVM", auc(fpr, tpr)))

# plt.figure(figsize=(10, 10))
# plt.plot([0, 1], [0, 1], 'k--')

# for model_name in history_dict:

csv_results_folder = "csv-results"
date_now = time.strftime("%Y-%m-%d")

auc_history = []
for k in range(0,3):
    auc_history.append(hypermodel[k].auc_history)

auc_history = np.array(auc_history)
if os.environ.get('OS','') == "Windows_NT":
    auc_history.tofile(f"{csv_results_folder}\evaluate_auc_{date_now}.csv", ",")
else:
    auc_history.tofile(f"{csv_results_folder}/evaluate_auc_{date_now}.csv", ",")

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend()
plt.show()
