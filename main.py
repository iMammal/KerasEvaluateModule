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
import keras_tuner as kt


class SequentialModel(kt.HyperModel):
    cvscores = []
    modelscores = []
    predictions = []
    history_dict = {}
    model = keras.Model
    name = "Sequential"
    def __init__(self, input_shape):
        self.input_shape = input_shape


    def build(self, hp):
        """Builds a Sequential model."""

        inputs = keras.Input(self.input_shape)    #shape=(28, 28, 1))
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(
            units=hp.Choice("units", [32, 64, 128]), activation="relu"
        )(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)

        # model.add(Dense(1, activation='sigmoid'))

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

        # model.compile(loss, optimizer, metrics)

    def modelevaluate(self, xtest, ytest, verbose=0):

        self.modelscores = model.evaluate(xtest, ytest, verbose=0)

        return self.modelscores

    def modelcvscores(self):
        print("%s: %.2f%%" % (model.metrics_names[1], self.modelscores[1] * 100))
        self.cvscores.append(self.modelscores[1] * 100)

        self.history_dict[model.name] = [self.history_callback, self.model]
        # model = history_dict[model_name][1]

    def modelpredict(self, xtest, verbose=0):
        self.predictions = model.predict(xtest)
        return self.predictions

    def printstats(self):
        print("%.2f%% (+/- %.2f%%)" % (np.mean(self.cvscores), np.std(self.cvscores)))


    # def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):
    #    super(kt.HyperModel,self).fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs)

    def fitmodel(self,X,Y,vd,cb):
        self.history_callback = model.fit(X, Y,
                  epochs=150,
                  batch_size=10,
                  verbose=0,
                  validation_data=vd,
                  callbacks=[cb])

        return self.history_callback

class SVMModel(kt.HyperModel):
    cvscores = []
    modelscores = []
    predictions = []
    history_dict = {}
    model = keras.Model
    name = "SVM"
    # model = SVC(kernel='linear')
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

    def printstats(self):
        # print("%.2f%% (+/- %.2f%%)" % (np.mean(self.cvscores), np.std(self.cvscores)))
        print("SVM stats...")
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
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# k = 0


hypermodel = {}


for train, test in kfold.split(X, Y):

    # modelDepth = k % 3
    # k += 1

    # create model
    inithp = kt.HyperParameters()

    hypermodel[0] = SVMModel()
    model = hypermodel[0].build(inithp)

    #if(modelDepth):
        # model = build_sequential_model()
    input_shape = (X[train].shape[1],)
        # (None, 28, 28, 1) #
    hypermodel[1] = SequentialModel(input_shape)
    model = hypermodel[1].build(inithp)
        # model.build()
        # Compile model
        # hypermodel.modelcompile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #else:

    # Fit the model
    #if(modelDepth):
    history_callback = hypermodel[1].fitmodel(X[train], Y[train],(X[test], Y[test]),cb)
    #else:
    hypermodel[0].fitmodel(X[train], Y[train])

    # evaluate the model
    #if(modelDepth):

    for k in range(0,2):
        hypermodel[k].modelevaluate(X[test], Y[test], verbose=0)

        hypermodel[k].modelcvscores()

        Y_pred = hypermodel[k].modelpredict(X[test])
        fpr, tpr, threshold = roc_curve(Y[test].ravel(), Y_pred.ravel())

    #if(modelDepth):
        plt.plot(fpr, tpr, 'k', label='{}, AUC = {:.3f}'.format(hypermodel[k].name, auc(fpr, tpr)))

        hypermodel[k].printstats()

    # else:
    #    print("SVM: %.2f",auc(fpr,tpr))
    #    plt.plot(fpr, tpr, 'k', label='{}, AUC = {:.3f}'.format("SVM", auc(fpr, tpr)))

# plt.figure(figsize=(10, 10))
# plt.plot([0, 1], [0, 1], 'k--')

# for model_name in history_dict:

plt.xlabel(                                                        'False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend()
plt.show()