import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Corrected import

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = 10  # Set to the correct number of classes for MNIST

# Reshape data to add a channel dimension
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Create an instance of ImageDataGenerator with data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,       # Randomly rotate images by 10 degrees
    width_shift_range=0.1,   # Shift images horizontally by 10%
    height_shift_range=0.1,  # Shift images vertically by 10%
    zoom_range=0.1,          # Random zoom on images
    shear_range=0.1          # Shear transformations
)

# Fit the ImageDataGenerator on the training data
datagen.fit(x_train)

# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Train the model using augmented data
batch_size = 128
epochs = 20
hist = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                 epochs=epochs,
                 validation_data=(x_test, y_test),
                 verbose=1)

# Save the model
model.save('mnist_augmented.h5')

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
