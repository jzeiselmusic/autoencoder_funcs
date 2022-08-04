import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from skimage import color
from keras.models import Sequential
from keras.layers import Activation, Input, Dropout, Reshape, Flatten, Dense
from keras.layers import Conv2D, UpSampling2D,AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

## model creator

def data_generator(batch_size, train_loc, val_loc):
  ## return a generator for training and validating samples
  ## shuffling is "true"
  ## my images are 270x480x3 color images

  BATCH_SIZE = batch_size
  train_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
  train_generator = train_datagen.flow_from_directory(
    train_loc,
    target_size=(270,480),
    batch_size=BATCH_SIZE,
    class_mode='input',
    shuffle=True
    )
  test_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
  validation_generator = test_datagen.flow_from_directory(
    val_loc,
    target_size=(270,480),
    batch_size=BATCH_SIZE,
    class_mode='input',
    shuffle= True
    )
  return train_generator, validation_generator

def testing_generator(batch_size_norm, batch_size_anom, test_loc, anom_loc):
  ## return a generator for testing normal and anomalous images
  ## shuffling is "false"
  test_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
  validation_generator = test_datagen.flow_from_directory(
    test_loc,
    target_size=(270,480),
    batch_size = batch_size_norm,
    class_mode='input',
    shuffle=False
    )

  anomaly_generator = test_datagen.flow_from_directory(
    anom_loc,
    target_size=(270,480),
    class_mode='input',
    batch_size = batch_size_anom,
    shuffle=False
    )
  return validation_generator, anomaly_generator

def build_model(FINAL_LAYER):
  ## returns encoder, decoder, and full autoencoder
  codings_size = [27,48,3]
  encoder_model = Sequential([Conv2D(3,(10,10),padding='same'), 
                        keras.layers.BatchNormalization(),
                        Activation('relu'),
                        AveragePooling2D(pool_size=(5,5), padding='same'),             
                        Activation('relu'),
                        AveragePooling2D(pool_size=(2,2),padding='same'),
                        Conv2D(3,(3,3),padding='same'),
                        keras.layers.BatchNormalization(),
                        Activation('relu'),
                        Flatten(),
                        Dense(FINAL_LAYER),
                        Activation('relu')])  

  decoder_model = Sequential([Dense(27*48*3),
                        Activation('relu'),
                        Reshape(codings_size),
                        Conv2D(3,(3,3),padding='same'),
                        keras.layers.BatchNormalization(),
                        Activation('relu'),       
                        UpSampling2D((2, 2)),   
                        Activation('relu'),
                        UpSampling2D((5,5)),
                        Conv2D(3,(10,10),padding='same'),
                        keras.layers.BatchNormalization(),
                        Activation('sigmoid')])
  inputs = keras.layers.Input(shape=(270,480,3))
  z = encoder_model(inputs)

  reconstructions = decoder_model(z)
  model = keras.Model(inputs=inputs,outputs=reconstructions)
  return model, encoder_model, decoder_model


def load_saved_model(name):
  model = tf.keras.models.load_model(name)
  return model

def save_model(model, name):
  model.save(name)
  print("model successfully saved to working directory")


def compile_train_model(model, EPOCHS, LR, B1, B2, train_generator, val_generator):
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.0, patience=3)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR,beta_1=B1,beta_2=B2),
               loss='binary_crossentropy')
  model.fit(
        train_generator,
        epochs=EPOCHS,
        callbacks=[callback],
        validation_data=val_generator)

  

def make_grey(img):
  return color.rgb2gray(img)
  

def visualize_results(model, num, generator):
  out_pred = model.predict(generator[num][0])[0,:,:,:]
  out_true = generator[num][0][0,:,:,:]

  plt.figure(0)
  plt.imshow(out_pred)
  plt.show()
  plt.figure(1)
  plt.imshow(out_true)
  plt.show()


def visualize_error(img_true,img_pred):
  error_map = abs(img_true - img_pred)
  plt.figure(0)
  plt.imshow(error_map)
  plt.show()

def calc_entropy(grey_error):
  # accepts a grey error plot
  # split the error plot up into 4 smaller sub-images
  # calculate the entropy in each one.
  # return the smallest entropy value
  i = int(grey_error.shape[0]/2)
  j = int(grey_error.shape[1]/2)
  ii = int(grey_error.shape[0])
  jj = int(grey_error.shape[1])

  error_plot_TL = error_plot[0:i,0:j]
  error_plot_TR = error_plot[0:i,j:jj]
  error_plot_BL = error_plot[i:ii,0:j]
  error_plot_BR = error_plot[i:ii,j:jj]

  hist_TL = np.histogram(error_plot_TL,bins=256)[1]
  hist_TR = np.histogram(error_plot_TR,bins=256)[1]
  hist_BL = np.histogram(error_plot_BL,bins=256)[1]
  hist_BR = np.histogram(error_plot_BR,bins=256)[1]

  ENT_TL = scipy.stats.entropy(hist_TL / np.sum(hist_TL))
  ENT_TR = scipy.stats.entropy(hist_TR / np.sum(hist_TR))
  ENT_BL = scipy.stats.entropy(hist_BL / np.sum(hist_BL))
  ENT_BR = scipy.stats.entropy(hist_BR / np.sum(hist_BR))

  return min([ENT_TL,ENT_TR,ENT_BL,ENT_BR])


def calc_std(grey_error):
  ## input a grey image that corresponds to an error plot
  ## find the basic 2d standard deviation in space of the error plot

  mean = np.mean(np.ndarray.flatten(grey_error))
  std = np.std(np.ndarray.flatten(grey_error))
  tuple_array = np.zeros((1,2))
  for i in range(grey_error.shape[0]):
      for j in range(grey_error.shape[1]):
          if (grey_error[i,j] > (mean+(2*std))):
              tuple_array = np.append(tuple_array,np.array([[i,j]]),0)
  tuple_array = np.delete(tuple_array,0,0) ## get rid of starter value

  total_std = np.std(tuple_array[:,0])*np.std(tuple_array[:,1])
  return total_std


def calc_mse(img, reconstruction, ORDER):
  ## calculate the mean error between the orig. color image and the reconstruction 
  ## order refers to the order of the mean error (mean squared (order=2), mean cubed (order=3), etc)
  error_map = abs(img - reconstruction) 
  mean_error = 10*(img - reconstruction)**ORDER
  mean_error = np.sum(mean_error,axis=None)
  mean_error = mean_error / (img.shape[0]*img.shape[1]*img.shape[2])
  return mean_error


def calc_total_error(img,reconstruction):
  ## calculate the mathematical total error between image and reconstruction
  error_map = abs(img - reconstruction)
  error_map = make_grey(error_map)
  total_error = np.sum(error_map)
  return total_error


def test_threshold(normal_array, anom_array, thresh, greater_or_less):
  ## print out the number of false positives, false negatives, etc
  ## within 2 arrays and a certain threshold, 
  ## must tell whether anomalies should be greater than or less than threshold ("<" or ">")
  import operator

  if greater_or_less == "<":
    op_func = operator.lt
    opp_op_func = operator.gt
  else:
    op_func = operator.gt
    opp_op_func = operator.lt

  CORRECT_NORMALS = 0
  INCORRECT_NORMALS = 0
  CORRECT_ANOMS = 0
  INCORRECT_ANOMS = 0
  for i in range(len(normal_array)):
    if opp_op_func(normal_array[i], thresh):
      CORRECT_NORMALS += 1
    else:
      INCORRECT_NORMALS += 1
  for i in range(len(anom_array)):
    if op_func(anom_array[i],thresh):
      CORRECT_ANOMS += 1
    else:
      INCORRECT_ANOMS += 1

  print(f"number of normals guessed correctly: {CORRECT_NORMALS}")
  print(f"number of normals guessed incorrectly: {INCORRECT_NORMALS}")
  print(f"number of anomalies guessed correctly: {CORRECT_ANOMS}")
  print(f"number of anomalies guessed incorrectly: {INCORRECT_ANOMS}")


def reconstruct(model, array, p):
  img = anom_img_array[p]
  reconstruction = model(np.expand_dims(img,0))
  reconstruction = np.squeeze(reconstruction)
  return reconstruction


def plot_hist(array_norm, array_anom, bins):
  ## plot one hist of two arrays
  ## blue is normal samples
  ## red is anomalous samples
  fig = plt.figure()
  plt.hist(array_norm,bins=bins,color='b')
  plt.hist(array_anom,bins=bins,color='r')
  plt.show()
