import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import AI_function

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
img_size = 128

# Now we can easily fetch our train and validation data.
train = AI_function.get_data('AI_model\Training_image')
val = AI_function.get_data('AI_model\Test_image')

lt = []
for i in train:
    lt.append(str(i[1]))

# Preprocessing
x_train, y_train, x_val, y_val = AI_function.preprocessing_data(train, val)

# Data augmnentation
datagen = ImageDataGenerator(
    featurewise_center=False,                   # set input mean to 0 over the dataset
    samplewise_center=False,                    # set each sample mean to 0
    featurewise_std_normalization=False,        # divide inputs by std of the dataset
    samplewise_std_normalization=False,         # divide each input by its std
    zca_whitening=False,                        # apply ZCA whitening
    rotation_range=30,                          # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.2,                             # Randomly zoom image
    width_shift_range=0.1,                      # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,                     # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,                       # randomly flip images
    vertical_flip=False)                        # randomly flip images

datagen.fit(x_train)

nb_epoch = 40

# Define model
model = AI_function.model_CNN()
history = model.fit(x_train, y_train, epochs=nb_epoch, validation_data=(x_val, y_val))

# Evaluating result
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(nb_epoch)

plt.figure(figsize=(15, 15))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Predictions
predictions = model.predict_classes(x_val)
predictions = predictions.reshape(1, -1)[0]
print(classification_report(y_val, predictions, target_names=labels))

AI_function.save_CNN_model(model, 'model_2.json', 'model_2.h5')
