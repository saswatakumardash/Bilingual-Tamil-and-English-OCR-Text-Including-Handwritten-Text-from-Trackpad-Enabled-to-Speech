import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Reshape, LSTM, GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------
# Step 1: Data Preprocessing
# -------------------------------------------

def load_dataset(dataset_dir, img_size=(64, 64)):
    X = []
    y = []
    for label in sorted(os.listdir(dataset_dir)):  # Loop through each folder (0 to 155)
        folder_path = os.path.join(dataset_dir, label)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".bmp"):  # Only process .BMP files
                    file_path = os.path.join(folder_path, file)
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
                    img = cv2.resize(img, img_size)  # Resize to 64x64 if needed
                    img = img / 255.0  # Normalize pixel values
                    X.append(img)
                    y.append(int(label))  # Label is the folder name (0 to 155)

    X = np.array(X).reshape(-1, img_size[0], img_size[1], 1)  # Reshape for CNN input
    y = np.array(y)
    y = to_categorical(y, num_classes=156)  # One-hot encode labels for 156 classes

    return X, y

# Load and preprocess the dataset
dataset_dir = "/Users/saswatakumardash/Desktop/Projects/Trackpad/tamil_data/dataset"  # Replace with your dataset path
X, y = load_dataset(dataset_dir)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------
# Step 2: Create CRNN Model with LSTM and GRU
# -------------------------------------------

def create_crnn_model(input_shape, num_classes):
    model = Sequential()

    # Convolutional layers for feature extraction
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    # Reshape the output to match LSTM/GRU input requirements
    model.add(Reshape((6 * 6, 128)))  # (6*6) time steps and 128 features

    # LSTM and GRU layers for sequence modeling
    model.add(LSTM(128, return_sequences=True))
    model.add(GRU(128, return_sequences=False))

    # Fully connected layers for classification
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Add dropout to prevent overfitting
    model.add(Dense(num_classes, activation='softmax'))

    return model

input_shape = (64, 64, 1)  # Image shape (64x64, grayscale)
num_classes = 156  # Total number of classes (characters)

# Create the model
crnn_model = create_crnn_model(input_shape, num_classes)

# Compile the model
crnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
crnn_model.summary()

# -------------------------------------------
# Step 3: Model Training and Saving
# -------------------------------------------

# Train the CRNN model
history = crnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = crnn_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model_name = "ocr_tamil.h5"
crnn_model.save(model_name)
print(f"Model saved as {model_name}")



# Predict on the test set
y_pred = crnn_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report for precision, recall, F1 score
print(classification_report(y_true, y_pred_classes, digits=4))

# -------------------------------------------
# Step 5: Bounding Box Visualization (Optional)
# -------------------------------------------

def draw_bounding_boxes(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours of characters
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display image with bounding boxes
    cv2.imshow("Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test bounding box visualization (optional)
# image_path = "path_to_test_image.bmp"
# draw_bounding_boxes(image_path)
