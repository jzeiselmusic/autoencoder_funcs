from autoencoder_funcs import *

BATCH_SIZE = 2
BATCH_ANOM = 80
BATCH_NORM = 650

## include image locations here ##
train_loc = ''
test_loc = ''
val_loc = ''
anom_loc = ''
##

train_generator, validation_generator = data_generator(BATCH_SIZE,train_loc,val_loc)
# test_generator, anom_generator = testing_generator(BATCH_NORM,BATCH_ANOM,test_loc,anom_loc)

anom_array = np.load('anomaly_images.npy') # load images into a numpy array
norm_array = np.load('normal_images.npy')  # because the generator is slow when testing

model, encoder, decoder = build_model(FINAL_LAYER=25)
compile_train_model(model, 20, .001, .8, .9, train_generator, validation_generator)
#model = load_saved_model('full_model_07_26_22')
#encoder = load_saved_model('encoder_model_07_26_22')
#decoder = load_saved_model('decoder_model_07_26_22')
#
## we want to test different anomaly criteria 

## first test total error 
print("testing total error...")

total_error_array_norms = []
for i in range(len(norm_array)):
  img = norm_array[i]
  rec = reconstruct(model, norm_array, i)
  total_error = calc_total_error(img, rec)
  total_error_array_norms.append(total_error)

total_error_array_anoms = []
for i in range(len(anom_array)):
  img = anom_array[i]
  rec = reconstruct(model, anom_array, i)
  total_error = calc_total_error(img, rec)
  total_error_array_anoms.append(total_error)

## now test mean squared error
print("testing total mse...")

ORDER = 4

mse_error_array_norms = []
for i in range(len(norm_array)):
  img = norm_array[i]
  rec = reconstruct(model, norm_array, i)
  mse_error = calc_mse(img, rec, ORDER)
  mse_error_array_norms.append(mse_error)

mse_error_array_anoms = []
for i in range(len(anom_array)):
  img = anom_array[i]
  rec = reconstruct(model, anom_array, i)
  mse_error = calc_mse(img, rec, ORDER)
  mse_error_array_anoms.append(mse_error)


## now test standard deviation plots
print("testing STD...")

std_array_norms = []
for i in range(len(norm_array)):
  img = norm_array[i]
  rec = reconstruct(model, norm_array, i)
  error = make_grey(abs(img - rec))  ## grey error plot
  total_std = calc_std(error)
  std_array_norms.append(total_std)

std_array_anoms = []
for i in range(len(anom_array)):
  img = anom_array[i]
  rec = reconstruct(model, anom_array, i)
  error = make_grey(abs(img - rec))  ## grey error plot
  total_std = calc_std(error)
  std_array_anoms.append(total_std)

## now test entropy 
print("testing entropy...")

entropy_array_norms = []
for i in range(len(norm_array)):
  img = norm_array[i]
  rec = reconstruct(model, norm_array, i)
  error = make_grey(abs(img - rec))  ## grey error plot
  entropy = calc_entropy(error, 256)
  entropy_array_norms.append(entropy)

entropy_array_anoms = []
for i in range(len(anom_array)):
  img = anom_array[i]
  rec = reconstruct(model, anom_array, i)
  error = make_grey(abs(img - rec))  ## grey error plot
  entropy = calc_entropy(error, 256)
  entropy_array_anoms.append(entropy)


## total error
plot_hist(total_error_array_norms, total_error_array_anoms, 100) # (norms < anoms)
## MSE
plot_hist(mse_error_array_norms, mse_error_array_anoms, 100) # (norms < anoms)
## STD
plot_hist(std_array_norms , std_array_anoms, 100) # (anoms < norms)
## entropy
plot_hist(entropy_array_norms , entropy_array_anoms, 100) # (anoms < norms)

##############################################################################
