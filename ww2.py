# prev: training acc 56, testing acc 34, time elapsed 38:48:23

import os

TF_ENABLE_ONEDNN_OPTS=0

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.initializers import HeUniform, RandomUniform
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import time 
from matplotlib import pyplot as plt

tf.config.optimizer.set_jit(True)
tf.random.set_seed(1)

IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
VAL_SPLIT = 0.15

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    rotation_range=40,
    zoom_range=0.3, 
    width_shift_range=0.3, 
    height_shift_range=0.3, 
    shear_range=0.3, 
    horizontal_flip=True, 
    brightness_range=[0.8, 1.2],
    validation_split=VAL_SPLIT,
)

train_gen = datagen.flow_from_directory(
    '../../ML_Data/CUB200/CUB_200_2011/images',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical', 
    subset='training',
)

val_gen = datagen.flow_from_directory(
    '../../ML_Data/CUB200/CUB_200_2011/images',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical', 
    subset='validation',
)

num_classes = len(train_gen.class_indices)

def build_model():
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(1024, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (3, 3), padding='same', kernel_initializer=HeUniform(), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(1024, kernel_initializer=RandomUniform(), activation='relu')(x)
    x = Dense(1024, kernel_initializer=RandomUniform(), activation='relu')(x)
    x = Dense(1024, kernel_initializer=RandomUniform(), activation='relu')(x)
    outputs = Dense(num_classes, kernel_initializer=RandomUniform(), activation='softmax')(x)

    model = Model(inputs, outputs)
    
    model.summary()
    
    model.compile(optimizer=Adam(learning_rate=0.0002, clipnorm=1.0), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = build_model()

#starting_epoch = int(input("What is the starting epoch?   "))

#model.load_weights("epoch_0"+str(starting_epoch)+".weights.h5")
#loss, acc = model.evaluate(val_gen, verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
#print("Restored model, loss: {:5.6f}".format(loss))

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('epoch_{epoch:03d}.weights.h5', save_weights_only=True, monitor='val_accuracy', save_best_only=False, save_freq=315*5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

start_time = time.time()

history = model.fit(
    train_gen,
    validation_data=val_gen,
    #initial_epoch=starting_epoch, 
    epochs=50,  # Adjust based on observation, with early stopping monitoring
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

end_time = time.time()

time_taken = end_time-start_time

hrs = int(time_taken/3600)
min = (int(time_taken/60))%60
sec = int(time_taken%60)

print(f"Time Elapsed:  {hrs} hours, {min} minutes, {sec} seconds.")
print(f"Time Elapsed:  {time_taken} seconds.")

_, train_acc = model.evaluate(train_gen, verbose=0)
_, test_acc = model.evaluate(val_gen, verbose=0)

print(f"Training Accuracy: {train_acc}, Testing Accuracy: {test_acc}")

# plot loss during training
plt.subplot(2, 1, 1)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training
plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()  

first_layer_weights = model.layers.get_weights()
