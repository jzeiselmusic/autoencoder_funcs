This repo is a library of functions that were helpful for me while creating and testing a convolutional autoencoder for image anomaly detection. 
The data generators search for images in google drive because that's where I stored all my images.
If you want to use these, you have to navigate to your own google drive with images located in a folder.

There are a number of anomaly criteria I tested out:
- total error 
- mean squared (or 4th) error
- amount of entropy in the error map images
- standard deviation of peak spots in error map images
