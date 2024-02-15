import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

import pandas as pd

need_correction = False

if(need_correction):
    # Load the CSV file
    df = pd.read_csv('scan_info.csv')

    # Correct the file paths
    df['file_path'] = df.apply(lambda row: row['file_path'].replace('scans/', ""), axis=1)
    df['mask_path'] = df.apply(lambda row: row['mask_path'].replace('masks/', ""), axis=1)

    # Save the csv file
    df.to_csv('scan_info.csv', index=False)

# Load the corrected file
scan_info = pd.read_csv('scan_info.csv')

# Convert in string 
scan_info['is_tumor'] = scan_info['is_tumor'].astype(str)
scan_info['is_tumor'] = scan_info['is_tumor'].map(lambda x: 'Sane' if x == '0' else 'Tumor')


# Define the directories
training_dir = 'train/train_scans/'
validation_dir = 'valid/valid_scans/'
test_dir = 'test/test_scans/'

num_classes = 2

# Load the pretrained model
image_size = [224, 224]
vgg = VGG16(input_shape=image_size + [3], weights='imagenet', include_top=False)

# Freeze the layers
for layer in vgg.layers:
    layer.trainable = False

# Add personalized layers
x = Flatten()(vgg.output)
x = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=x)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


batch_size = 32

# Match labels to file paths
training_df = scan_info[scan_info['division'] == 'train']
validation_df = scan_info[scan_info['division'] == 'valid']
test_df = scan_info[scan_info['division'] == 'test']

training_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
validation_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)

training_generator = training_datagen.flow_from_dataframe(dataframe=training_df,
                                                          directory=training_dir,
                                                          x_col='file_path',
                                                          y_col='is_tumor',
                                                          target_size=image_size,
                                                          batch_size=batch_size,
                                                          class_mode='categorical')

validation_generator = validation_datagen.flow_from_dataframe(dataframe=validation_df,
                                                              directory=validation_dir,
                                                              x_col='file_path',
                                                              y_col='is_tumor',
                                                              target_size=image_size,
                                                              batch_size=batch_size,
                                                              class_mode='categorical')

test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                  directory=test_dir,
                                                  x_col='file_path',
                                                  y_col='is_tumor',
                                                  target_size=image_size,
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

print(training_generator.class_indices)

train_images = len(scan_info[scan_info['division'] == 'train'])
valid_images = len(scan_info[scan_info['division'] == 'valid'])
test_images = len(scan_info[scan_info['division'] == 'test'])


history = model.fit(training_generator,
                    steps_per_epoch = train_images // batch_size,  
                    epochs = 1,  # change this for better results
                    validation_data = validation_generator,
                    validation_steps = valid_images // batch_size)  

print ('Training Accuracy = ' + str(history.history['accuracy']))
print ('Validation Accuracy = ' + str(history.history['val_accuracy']))

# Evaluate the model on the test set
test_images = len(scan_info[scan_info['division'] == 'test'])
test_results = model.evaluate(test_generator, 
                              steps = test_images // batch_size)

# Plot training and validation metrics
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Metrics')
plt.xlabel('Epochs')
plt.show()

# Generate confusion matrix for the test set
from sklearn.metrics import confusion_matrix, classification_report

# Predict on the test set
test_predictions = model.predict(test_generator, 
                                 steps=test_images // batch_size + 1)
test_predictions = np.argmax(test_predictions, 
                             axis=1)

# Get the true labels
test_true_labels = test_generator.classes

# Generate confusion matrix
conf_matrix = confusion_matrix(test_true_labels, 
                               test_predictions)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
print("Classification Report:")
print(classification_report(test_true_labels, 
                            test_predictions))
