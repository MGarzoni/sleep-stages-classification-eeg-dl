# IMPORT LIBRARIES AND DEPENDENCIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import keras
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split

# FUNCTION TO PERFORM SHORT FAST FOURIER TRANSFORM ON RAW EEG DATA FILE TO COMPUTE THE SPECTROGRAM
def create_spectrograms(data):
    
    # input: eeg waves, inputs
    # output: spectrogram of eeg wave, transformed inputs
    
    spectograms = []
    for d in data:
        _, _, Sxx = spectrogram(d, 100, nperseg=200, noverlap=105)
        Sxx = Sxx[:, 1:,:]
        spectograms.append(Sxx)
    return np.array(spectograms)

# READ IN DATASET AND UPDATE X:INPUTS IN THE SPECTROGRAM DATA TO USE AS INPUTS FOR THE MODELS
DATAPATH = 'C:\\Users\\Eigenaar\\Desktop\\Final Semester\\DL\\Assignment\\Data_Raw_signals.pkl'
x, labels = pd.read_pickle(DATAPATH)
inputs = np.array(spectrogram(x, 100, nperseg=200, noverlap=105)[2], dtype = "float32")

# SIMPLE TRAIN/TEST SPLIT ON DATASET
inputs_train, inputs_test, \
labels_train, labels_test = train_test_split(inputs, labels, test_size=0.25)

##################### PREPROCESSING AND VISUALIZATION
# check for class imbalance    
with open(DATAPATH, "rb") as fp:
    buffer = pickle.load(fp)
    labels = buffer[1]
    df = pd.DataFrame(buffer[0][0][0])
    print(buffer[0][123])
    plt.plot(df)
    
    '''
    Checking if the observations for each labels are balanced.
    We see that there is not a great difference between the classes
    '''
    
    REM_stage_label_count = 0
    stage1_label_count = 0
    stage2_label_count = 0
    stage3_label_count = 0
    stage4_label_count = 0
    waking_stage_label_count = 0
    
    for l in labels:
        if l == 0:
            REM_stage_label_count += 1
        if l == 1:
            stage1_label_count += 1
        if l == 2:
            stage2_label_count += 1
        if l == 3:
            stage3_label_count += 1
        if l == 4:
            stage4_label_count += 1
        if l == 5:
            waking_stage_label_count += 1
            
    print("Count of label 0 (REM stage): {},\n Count of label 1: {},\n\
          Count of label 2: {},\n Count of label 3: {},\n Count of label 4: {},\n\
          Count of label 5 (Waking stage): {}".format(REM_stage_label_count, stage1_label_count,\
          stage2_label_count, stage3_label_count, stage4_label_count, waking_stage_label_count))

# visualize EEG waves
fig_1 = plt.figure(figsize=(10, 5))
plt.plot(inputs[100, ...].ravel())
plt.title("EEG Epoch")
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.show()

# visualize sleep stages
fig_2 = plt.figure(figsize=(10, 5))
plt.plot(labels.ravel())
plt.title("Sleep Stages")
plt.ylabel("Classes")
plt.xlabel("Time")
plt.show()

############################### MODELS
# tryout with a 
# sequential model with different layers
from keras.models import Sequential
from keras import layers

num_classes = 6 # number of output units

model = Sequential() # initialize sequential model
model.add(layers.Conv2D(filters=6, kernel_size=(1,2),
                        kernel_initializer="glorot_uniform",
                        kernel_constraint=None))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(filters=4, kernel_size=(1,2), strides=(2, 2), activation="relu",
                        kernel_initializer="glorot_uniform",
                        bias_initializer="glorot_uniform",
                        kernel_constraint=None,
                        bias_constraint=None))
model.add(layers.SpatialDropout2D(rate=0.05)) # add a spatial dropout layer
model.add(layers.Flatten()) # flatten tensor from 4D to 2D so Dense layers can deal with it
for i in range(5):
    model.add(layers.Dense(72, activation="tanh")) # add 10 dense layers
    model.add(keras.layers.Dropout(rate=0.15))

model.add(keras.layers.Dense(num_classes, activation="softmax")) # output layer

model.compile(optimizer = "rmsprop", 
                loss = keras.losses.SparseCategoricalCrossentropy(),
                metrics = ["accuracy"])

# fit the model to the train data and evaluate it on the test data
model.fit(inputs_train, labels_train, epochs=25)
names = model.metrics_names
test_scores = model.evaluate(inputs_test, labels_test)
print("\n")
for n, ts in zip(names, test_scores):
    print(f"{n}: {ts}")

# save model for later
model.save("best_for_now.h5")

#########################################################################################
#########################################################################################


# SIMPLE TRAIN/TEST SPLIT ON DATASET
inputs_train, inputs_test, \
labels_train, labels_test = train_test_split(x, labels, test_size=0.25)

keras.backend.clear_session() # clear model data

# create and fit the LSTM network
rnn = Sequential()
rnn.add(layers.GRU(50))
for i in range(5):
    rnn.add(layers.Dense(64, activation="relu")) # add 10 dense layers
    rnn.add(keras.layers.Dropout(rate=0.05))
rnn.add(layers.Dense(6, activation="softmax")) # output layer

rnn.compile(optimizer = "rmsprop", 
                loss = keras.losses.SparseCategoricalCrossentropy(),
                metrics = ["accuracy"])

# fit the model to the train data and evaluate it on the test data
rnn.fit(inputs_train, labels_train, epochs=10)
names = rnn.metrics_names
test_scores = rnn.evaluate(inputs_test, labels_test)
print("\n")
for n, ts in zip(names, test_scores):
    print(f"{n}: {ts}")


#########################################################################################
#########################################################################################

# load model
from keras.models import load_model
model = load_model('best_for_now.h5')

# summarize model
model.summary()

mainpath = 'Data_Raw_signals.pkl'

#test_raw_signals_no_labels = test data for which we have to find the labels, and submit these labels to codaLab
testpath = "Test_Raw_signals_no_labels.pkl"

#x, y = pd.read_pickle(mainpath)
#xx = pd.read_pickle(testpath)
#
#x = np.array(spectrogram(x, 100, nperseg=200, noverlap=105)[2], dtype = "float32")
#xx = x[:,1,:,:]
#
#train_x, dev_x, train_y, dev_y = train_test_split(x, y, test_size=0.2)
#test_x = train_x
#test_y = train_y

x = pd.read_pickle(testpath)[0]
x = np.array(spectrogram(x, 100, nperseg=200, noverlap=105)[2], dtype = "float32")

def getPredictions(x, model):
    predictions = []
    tmp = model.predict(x)
    tmp = np.argmax(tmp, axis = 1)
    predictions.append(tmp)
    predictions = np.transpose(np.array(predictions))
    return predictions
    
pred_labels = getPredictions(x, model)
pred_labels = [int(l) for l in pred_labels]
with open("ANSWERSRS.txt", "w") as fh:
    for l in pred_labels:
        fh.write(str(l))
        fh.write("\n")


