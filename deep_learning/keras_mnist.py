from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False, help="path to the output loss/accurancy plot")
args = vars(ap.parse_args)

#Grab the MINST dataset
print("Loading dataset")
dataset = datasets.fetch_openml('mnist_784', version=1)

#Scale the raw pixel intensenties to the range [0, 1.0],
#then construct the trainina and testing splits
data = dataset.data.astype("float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size = 0.25)

#Convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#Define the 784-256-128-10 architecture
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

#train model using SGD
print("Training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

#Evalute the network
print("Evaluating network..")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

#plot the training loss and accurancy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label="train_loss")
plt.plot(np.arange(0, 100), H.history['val_loss'], label="val_loss")
plt.plot(np.arange(0, 100), H.history['accuracy'], label="train_acc")
plt.plot(np.arange(0, 100), H.history['val_accuracy'], label="val_acc")
plt.title("Training Loss and Accurancy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accurancy")
plt.legend()
plt.show()
#plt.savefig(args['output'])
