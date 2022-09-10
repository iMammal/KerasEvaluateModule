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

        return keras.Model(inputs=inputs, outputs=outputs)


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

history_dict = {}

# TensorBoard Callback
cb = TensorBoard()

# load Gencode  dataset
dataset = np.loadtxt("NewGencode4DLTraining.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:, 0:5]
Y = dataset[:, 5]

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

k = 0





for train, test in kfold.split(X, Y):

    modelDepth = k % 3
    k = k + 1

    # create model

    if(modelDepth):
        # model = build_sequential_model()
        input_shape = (X[train].shape[1],)
        # (None, 28, 28, 1) #
        hypermodel = SequentialModel(input_shape)
        inithp = kt.HyperParameters()
        model = hypermodel.build(inithp)
        # model.build()
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    else:
        model=SVC(kernel='linear')

    # Fit the model
    if(modelDepth):
        history_callback = model.fit(X[train], Y[train],
                                 epochs=150,
                                 batch_size=10,
                                 verbose=0,
                                 validation_data=(X[test], Y[test]),
                                 callbacks=[cb])
    else:
        model.fit(X[train], Y[train])

    # evaluate the model
    if(modelDepth):
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)

        history_dict[model.name] = [history_callback, model]
        # model = history_dict[model_name][1]


    Y_pred = model.predict(X[test])
    fpr, tpr, threshold = roc_curve(Y[test].ravel(), Y_pred.ravel())

    if(modelDepth):
        plt.plot(fpr, tpr, 'k', fpr, tpr, 'bo', label='{}, AUC = {:.3f}'.format(model.name, auc(fpr, tpr)))

        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    else:
        print("SVM: %.2f",auc(fpr,tpr))
        plt.plot(fpr, tpr, 'k', fpr, tpr, 'bo', label='{}, AUC = {:.3f}'.format("SVM", auc(fpr, tpr)))

# plt.figure(figsize=(10, 10))
# plt.plot([0, 1], [0, 1], 'k--')

# for model_name in history_dict:

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend()
plt.show()