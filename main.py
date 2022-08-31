from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

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

for train, test in kfold.split(X, Y):
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=5, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # model.name = "Dense2Layers"

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    history_callback = model.fit(X[train], Y[train],
                                 epochs=150,
                                 batch_size=10,
                                 verbose=0,
                                 validation_data=(X[test], Y[test]),
                                 callbacks=[cb])

    # evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

    history_dict[model.name] = [history_callback, model]
    # model = history_dict[model_name][1]

    Y_pred = model.predict(X[test])
    fpr, tpr, threshold = roc_curve(Y[test].ravel(), Y_pred.ravel())

    plt.plot(fpr, tpr, label='{}, AUC = {:.3f}'.format(model.name, auc(fpr, tpr)))

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# plt.figure(figsize=(10, 10))
# plt.plot([0, 1], [0, 1], 'k--')

# for model_name in history_dict:

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
# plt.legend()
plt.show()