import pandas as pd
import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Load the CSV data file
df = pd.read_csv('scan_info.csv')

# Create the label column C
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['is_tumor'])

# Define the directory paths 
train_path = 'train/train_scans/'
train_mask_path = 'train/train_masks/'

val_path = 'valid/valid_scans/'
val_mask_path = 'valid/valid_masks/'

test_path = 'test/test_scans/'
test_mask_path = 'test/test_masks/'

# Masks and scans loading function
def load_images_and_masks(dataframe, image_path, mask_path):
    images = []
    masks = []
    labels = []

    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        image_filepath = os.path.join(image_path, row['file_path'].split('scans/', 1)[-1])
        mask_filepath = os.path.join(mask_path, row['mask_path'].split('masks/', 1)[-1])

        image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Error loading images for image: {image_filepath}")
            continue

        # Images resizing
        image = cv2.resize(image, (224, 224))
        mask = cv2.resize(mask, (224, 224))

        images.append(image)
        masks.append(mask)
        labels.append(row['label'])

    return np.array(images), np.array(masks), np.array(labels)

# Load the data
train_images, train_masks, train_labels = load_images_and_masks(df[df['division'] == 'train'], train_path, train_mask_path)
val_images, val_masks, val_labels = load_images_and_masks(df[df['division'] == 'valid'], val_path, val_mask_path)
test_images, test_masks, test_labels = load_images_and_masks(df[df['division'] == 'test'], test_path, test_mask_path)

# Normalize images and masks
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

train_masks = train_masks / 255.0
val_masks = val_masks / 255.0
test_masks = test_masks / 255.0

# CNN model
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Model compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training with data augmentation
train_datagen = ImageDataGenerator(rotation_range=20, 
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True, 
                                   fill_mode='nearest')

train_generator = train_datagen.flow(train_images.reshape(-1, 224, 224, 1), 
                                     train_labels, 
                                     batch_size=32)

history = model.fit(train_generator, 
                    epochs=50, 
                    validation_data=(val_images.reshape(-1, 224, 224, 1), val_labels),
                    verbose=1)

print('history is made\n')

# Évaluation du modèle sur l'ensemble de test
test_loss, test_accuracy = model.evaluate(test_images.reshape(-1, 224, 224, 1), test_labels)
print(f'Test Accuracy: {test_accuracy}')

train_loss, train_accuracy = model.evaluate(train_images.reshape(-1, 224, 224, 1), train_labels)
print(f'Train Accuracy: {train_accuracy}')

# Sauvegarde du modèle
model.save('liver_disease_classifier.keras')

def plot_metrics(history, save_path=None):
    plt.figure(figsize=(12, 8))

    # Précision
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Perte
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if save_path:
        plt.savefig(os.path.join(save_path, 'metrics_plot.png'))
        plt.show()
    else:
        plt.show()


# Fonction pour afficher la matrice de confusion
def plot_confusion_matrix(y_true, y_pred, classes, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')

    if save_path:
        plt.savefig(os.path.join(save_path, title))
        plt.show()
    else:
        plt.show()

# Prédictions sur l'ensemble de test
test_predictions = model.predict(test_images.reshape(-1, 224, 224, 1))
test_predictions = (test_predictions > 0.5).astype(int)

train_predictions = model.predict(train_images.reshape(-1, 224, 224, 1))
train_predictions = (train_predictions > 0.5).astype(int)
# Affichage des courbes de métriques

test_loss, test_accuracy = model.evaluate(test_images.reshape(-1, 224, 224, 1), test_labels)
print(f'Test Accuracy: {test_accuracy}')


# Affichage des courbes de métriques
plot_metrics(history, save_path=r'C:\Users\yanis\OneDrive\Documents\projet_reseaux_neurones')

# Affichage de la matrice de confusion
plot_confusion_matrix(test_labels, test_predictions, classes=['Foie Sain', 'Foie Malade'], title='test_confusion_matrix.png', save_path=r'C:\Users\yanis\OneDrive\Documents\projet_reseaux_neurones')

plot_confusion_matrix(train_labels, train_predictions, classes=['Foie Sain', 'Foie Malade'], title='train_confusion_matrix.png', save_path=r'C:\Users\yanis\OneDrive\Documents\projet_reseaux_neurones')

# Rapport de classification
print("Classification Report:\n", classification_report(test_labels, test_predictions))

print("Classification Report (Training Set):\n", classification_report(train_labels, train_predictions))
