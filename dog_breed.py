# Ideas taken from https://www.kaggle.com/gaborfodor/dog-breed-pretrained-keras-models-lb-0-3
import cv2
import numpy
import numpy as np
import pandas as pd
from keras import backend as k
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
import sys

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
numpy.random.seed(seed)
# Number of classes
num_classes = y_test.shape[1]

print("The number of classes is {}".format(num_classes))

# Idea from https://keras.io/applications/
if trained_model == "inception":
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(im_size, im_size, 3))

elif trained_model == "vgg16":
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(im_size, im_size, 3))

elif trained_model == "resnet50":
    base_model =  ResNet50(weights='imagenet', include_top=False, input_shape=(im_size, im_size, 3))

elif trained_model == "mobilenet":
    base_model = MobileNet(input_shape=(im_size, im_size, 3), include_top=False, weights='imagenet')

else:
    sys.exit(" Could not find the train model to start with")
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer --
predictions = Dense(num_class, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy')


# Function came from https://gist.github.com/Hironsan/e041d6606164bc14c50aa56b989c5fc0
def batch_iter(data, labels, batch_size_def, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size_def) + 1

    def data_generator():
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size_def
                end_index = min((batch_num + 1) * batch_size_def, data_size)
                x_value, y_value = shuffled_data[start_index: end_index], shuffled_labels[start_index: end_index]
                yield x_value, y_value

    return num_batches_per_epoch, data_generator()


# Code idea came from https://gist.github.com/Hironsan/e041d6606164bc14c50aa56b989c5fc0


batch_size = 32
num_epochs = 2

train_steps, train_batches = batch_iter(X_train, y_train, batch_size)
valid_steps, valid_batches = batch_iter(X_test, y_test, batch_size)
model.fit_generator(train_batches, train_steps, epochs=num_epochs, validation_data=valid_batches,
                    validation_steps=valid_steps)

save_model = False
if save_model:
# serialize model to JSON
    model_json = model.to_json()
    with open("{}/{}_model.json".format(WORKING_DIR,trained_model), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("{}/{}_model.h5".format(WORKING_DIR,trained_model))
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
sub.to_csv("{}/results_{}.csv".format(WORKING_DIR,trained_model), mode='w', index=False)
