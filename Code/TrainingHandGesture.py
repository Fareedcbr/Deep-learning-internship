from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os
import time

# Define dataset paths
train_path = os.path.join('Internship on AI', 'Day 13 - Hand Gesture Recognition using DL', 'HandGestureDataset', 'train')
test_path = os.path.join('Internship on AI', 'Day 13 - Hand Gesture Recognition using DL', 'HandGestureDataset', 'test')

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=150, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=6, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=12.,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.15,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(256, 256),
                                                 color_mode='grayscale',
                                                 batch_size=8,
                                                 classes=['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                                 class_mode='categorical')

val_set = val_datagen.flow_from_directory(test_path,
                                          target_size=(256, 256),
                                          color_mode='grayscale',
                                          batch_size=8,
                                          classes=['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                          class_mode='categorical')

# Define callbacks
model_filename = f"model_{int(time.time())}.h5"
callback_list = [
    EarlyStopping(monitor='val_loss', patience=10),
    ModelCheckpoint(filepath=model_filename, monitor='val_loss', save_best_only=True, verbose=1)
]

# Train the model
model.fit(training_set,
          steps_per_epoch=len(training_set) // training_set.batch_size,
          epochs=25,
          validation_data=val_set,
          validation_steps=len(val_set) // val_set.batch_size,
          callbacks=callback_list)