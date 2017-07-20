# import training data
import pandas as pd
import cv2
import matplotlib.pyplot as plt

log_file = pd.read_csv('driving_log.csv',
                       names = ["center_img", "left_img", "right_img", "steer", 
                                "throttle", "break", "speed"])

# preview the dataset
#print(log_file.head(10))
log_file.steer.hist(bins = 51)
plt.xlabel("Steer")
plt.ylabel("Count")
plt.xlim([-1.0, 1.0])
plt.ylim([0,200])
plt.title("Steering angle distribution")
plt.show()

# give lower weight to images with 0 steer
log_file["weight"] = 1.0
log_file.loc[log_file["steer"] == 0.0, "weight"] = 0.1

def load_img(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# test preprocess with one image
image = load_img(log_file.loc[0, "center_img"])
cropped = image[50:, :, :]
resized = cv2.resize(cropped, (200, 66))
flipped = cv2.flip(resized, 1)
plt.imshow(flipped)
plt.show()
print('steer is ', log_file.loc[0, 'steer'])

# data generator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

train_idx, valid_idx = train_test_split(log_file.index.values, test_size=0.3)
#train_idx, valid_idx = train_test_split(log_file[log_file["steer"] != 0.0].index.values, test_size=0.3)

# generator 
def data_generator(idx, batch_size = 128):
    '''
    load center, left, and right images as X
    steer + correction as y
    weight from pre-calculated values
    '''
    n_samples = len(idx) * 3 # 3 images per line
    steer_correction = 0.1 # 0.1 or 0.05 works
    image_filenames = log_file.loc[idx, 
                ["center_img", "left_img", "right_img"]].values.flatten("F") # column first flatten
    labels = log_file.loc[idx, "steer"].values
    labels = np.append(labels, [labels + steer_correction, labels - steer_correction])
    weights = log_file.loc[idx, "weight"].values
    weights = np.append(weights, [weights, weights])
    
    while 1:
        image_filenames, labels, weights = shuffle(image_filenames, labels, weights) # shuffle each epoch
        for offset in range(0, n_samples, batch_size):
            X_values = []
            y_values = []
            
            for image_filename, label, flip in zip(image_filenames[offset:offset+batch_size],
                                                   labels[offset:offset+batch_size],
                                                   np.random.randint(2,size=batch_size)):
                image = load_img(image_filename)
                if flip:
                    # randomly flip images 50/50
                    image = cv2.flip(image,1)
                    label = -label
                X_values.append(image)
                y_values.append(label)
                
            X_values = np.array(X_values)
            y_values = np.array(y_values)
            w_values = weights[offset:offset+batch_size]
            yield shuffle(X_values, y_values, w_values)

batch_size = 128
train_generator = data_generator(train_idx, batch_size = batch_size)
valid_generator = data_generator(valid_idx, batch_size = batch_size)

# verify batch generator function
#steps_per_epoch = (len(train_idx) * 3 - 1) // batch_size + 1            
#for i in range(steps_per_epoch):
#    X_batch, y_batch = next(train_generator)
#    print(X_batch.shape, y_batch.shape)

# build the model
from keras.layers import Lambda, Cropping2D, Flatten, Dense, Dropout
from keras.models import Model, Sequential, load_model
from keras.layers.convolutional import Conv2D
from pathlib import Path

def resize_normalize(image):  
    '''
    resize to 66 x 200 image (as in Nvidia model)
    also normailize pixel to (-0.5, 0.5)
    need to explicitly define lambda function to reload model
    '''
    from keras.backend import tf as ktf
    resized = ktf.image.resize_images(image, (66, 200))
    normalized = resized / 255.0 - 0.5
    return normalized

def SimpleModel():
    '''
    simple model to test correctness of cropping and resize etc
    '''
    simple_model = Sequential()
    simple_model.add(Cropping2D(cropping=((50,0),(0,0)), input_shape=(160, 320, 3))) # cropping
    simple_model.add(Lambda(resize_normalize)) # resizing
    simple_model.add(Flatten())
    simple_model.add(Dense(1))
    simple_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return simple_model

def NvidiaNet():
    '''
    CNN using the Nvidia arch
    '''
    nvidia_net = Sequential()
    nvidia_net.add(Cropping2D(cropping=((50,0),(0,0)), input_shape=(160, 320, 3))) # cropping
    nvidia_net.add(Lambda(resize_normalize)) # resizing
    nvidia_net.add(Conv2D(24, 5,5, subsample=(2,2), activation = 'relu'))
    nvidia_net.add(Conv2D(36, 5,5, subsample=(2,2), activation = 'relu'))
    nvidia_net.add(Conv2D(48, 5,5, subsample=(2,2), activation = 'relu'))
    nvidia_net.add(Conv2D(64, 3,3, activation = 'relu'))
    nvidia_net.add(Conv2D(64, 3,3, activation = 'relu'))
    nvidia_net.add(Flatten())
    nvidia_net.add(Dropout(0.5))
    nvidia_net.add(Dense(100, activation='relu'))
    nvidia_net.add(Dropout(0.3))
    nvidia_net.add(Dense(50, activation='relu'))
    nvidia_net.add(Dropout(0.3))
    nvidia_net.add(Dense(10, activation='relu'))
    nvidia_net.add(Dense(1))
    nvidia_net.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return nvidia_net

model_file = Path('model.h5')
if model_file.is_file():
    model = load_model('model.h5')
else:
    model = NvidiaNet()

# fit the data
history_object = model.fit_generator(train_generator, len(train_idx) * 3, nb_epoch = 5,
                           validation_data=valid_generator, nb_val_samples=len(valid_idx)*3)

model.save('model.h5')

### plot the training and validation loss for each epoch
plt.figure()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()