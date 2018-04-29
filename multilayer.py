import cv2
import numpy as np
import pandas as pd
from keras import backend as k
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tqdm import tqdm

trained_model = "vgg16"  # inception, vgg16, resnet50, mobilenet

#  Get working directory and
WORKING_DIR = "./data/"
# Location of labels
LABELS = WORKING_DIR + "labels.csv"
# Example of the submission text
TEST = WORKING_DIR + "sample_submission.csv"

# Location of train and test folders
TRAIN_FOLDER = WORKING_DIR + "/train/"
TEST_FOLDER = WORKING_DIR + "/test/"

# Read in the labels and the test data
df_train = pd.read_csv(LABELS)
df_test = pd.read_csv(TEST)

targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse=True)
one_hot_labels = np.asarray(one_hot)

# Deisred image size
im_size = 299
# Number of classifications
x_train_0 = []
y_train_0 = []
x_test_0 = []

i = 0
# Getting the training data
for f, breed in tqdm(df_train.values):
    img = cv2.imread('{}{}.jpg'.format(TRAIN_FOLDER, f))
    label = one_hot_labels[i]
    x_train_0.append(img)
    y_train_0.append(label)
    i += 1

# Get the data used for the kaggle compititon results
for f in tqdm(df_test['id'].values):
    img = cv2.imread('{}{}.jpg'.format(TEST_FOLDER, f))
    x_test_0.append(img)

# 0 to 1 instead of 255 collars

y_train_raw = y_train_0
x_train_raw = np.divide(x_train_0, 255.)
x_test = np.divide(x_test_0, 255.)

# Check shape
print(x_train_raw.shape)
print(y_train_raw.shape)
print(x_test.shape)

num_class = y_train_raw.shape[1]

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(x_train_raw, y_train_raw, test_size=0.2, random_state=1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Implement Batch Normalization


k.set_image_dim_ordering('tf')

# fix random seed for reproducibility
seed = 123
np.random.seed(seed)
# Number of classes
num_classes = y_test.shape[1]


def multilayer_cnn_model():
    # YOUR TURN
    # Build a model with 4 convolutional layers
    # choose your own hyperparameters for conv layers
    # choose to include maxpool if you like
    # choose to include dropout if you like
    # create model
    inner_model = Sequential()

    # Add 32 filters
    inner_model.add(Conv2D(64, kernel_size=3, padding='same',
                           input_shape=(im_size, im_size, 3)))
    inner_model.add(Activation('relu'))

    # Conv2D 32 3 x3
    inner_model.add(Conv2D(64, kernel_size=3, padding='same'))
    inner_model.add(Activation('relu'))
    # 2x2 pooling
    inner_model.add(MaxPooling2D((2, 2)))
    # Conv2D 64  3x3
    inner_model.add(Conv2D(64, kernel_size=3, padding='same'))
    inner_model.add(Activation('relu'))
    # Conv 3D 8x8
    inner_model.add(Conv2D(64, kernel_size=3, padding='same'))
    inner_model.add(Activation('relu'))
    # 2x2 pooling
    inner_model.add(MaxPooling2D((2, 2)))
    # Flatten
    inner_model.add(Flatten())

    # . Density layer
    inner_model.add(Dense(512, activation='relu'))
    # . Density layer
    inner_model.add(Dense(512, activation='relu'))
    #     # output layer
    inner_model.add(Dense(num_classes, activation='softmax'))
    inner_model.summary()
    # Compile model using the same options
    inner_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return inner_model


def bn_model():
    # YOUR TURN
    # Create a model with 4 convolutional layers (2 repeating VGG stype units) and 2 dense layers before the output
    # Use Batch Normalization for every conv and dense layers
    # Use dropout layers if you like
    # Use Adam optimizer

    model = Sequential()
    model.add(Conv2D(16, 3, input_shape=(im_size, im_size, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model

train_model = "multilayer"
if train_model == "multilayer":
    model = multilayer_cnn_model()
else:
    model = bn_model()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200,
                    verbose=2)  # Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

# serialize model to JSON
model_json = model.to_json()
with open("{}/{}_model.json".format(WORKING_DIR, trained_model), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("{}/{}_model.h5".format(WORKING_DIR, trained_model))
print("Saved model to disk")

preds = model.predict(x_test, verbose=1)

sub = pd.DataFrame(preds)
# Set column names to those generated by the one-hot encoding earlier
col_names = one_hot.columns.values
sub.columns = col_names
# Insert the column id from the sample_submission at the start of the data frame
sub.insert(0, 'id', df_test['id'])
sub.head(5)

# Saving results
sub.to_csv("{}/results_{}.csv".format(WORKING_DIR, trained_model), mode='w', index=False)
