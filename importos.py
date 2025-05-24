import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import datetime

def preprocess_image(image, image_size=(64, 64)):
    image = cv2.resize(image, image_size)
    image = image / 255.0
    return image

def load_dataset(dataset_path, image_size=(64, 64)):
    images = []
    labels = []

    print(f"Loading dataset from: {dataset_path}")

    for label in os.listdir(dataset_path):
        label_folder = os.path.join(dataset_path, label)
        if os.path.isdir(label_folder):
            print(f"Processing label: {label}")
            for image_name in os.listdir(label_folder):
                image_path = os.path.join(label_folder, image_name)
                if image_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Warning: Unable to load image: {image_path}")
                        continue
                    image = preprocess_image(image, image_size)
                    images.append(image)
                    labels.append(label)

    if not images:
        raise ValueError("No images were loaded. Check the dataset path and file extensions.")

    images = np.array(images)
    labels = np.array(labels)

    label_binarizer = LabelBinarizer()
    if len(labels) > 0:
        labels = label_binarizer.fit_transform(labels)
    else:
        raise ValueError("No labels found. Check the dataset path and structure.")

    return images, labels, label_binarizer

def create_model(input_shape=(64, 64, 3), num_classes=28):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def test_with_camera(model, label_binarizer):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow("Sign Language Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't capture frame.")
            break

        display_frame = frame.copy()
        height, width = frame.shape[:2]
        box_size = 300
        top_left = (width // 2 - box_size // 2, height // 2 - box_size // 2)
        bottom_right = (width // 2 + box_size // 2, height // 2 + box_size // 2)

       
        cv2.rectangle(display_frame, top_left, bottom_right, (0, 255, 0), 2)

 
        roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        processed_roi = preprocess_image(roi)
        input_tensor = np.expand_dims(processed_roi, axis=0)


        predictions = model.predict(input_tensor, verbose=0)
        predicted_class = np.argmax(predictions[0])
        current_prediction = label_binarizer.classes_[predicted_class]


        cv2.putText(display_frame, f"Sign: {current_prediction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Place hand in green box", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Sign Language Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    dataset_path = r'C:\Users\SADOK\Desktop\ahmed\csv\ALPHA'  

    try:
        print("Loading dataset...")
        images, labels, label_binarizer = load_dataset(dataset_path)
        print(f"\nDataset loaded successfully!")
        print(f"Total images: {len(images)}")
        print(f"Number of classes: {len(label_binarizer.classes_)}")
    except ValueError as e:
        print(f"Error: {e}")
        exit()

    print("\nSplitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nCreating model...")
    model = create_model(
        input_shape=(64, 64, 3), 
        num_classes=len(label_binarizer.classes_)
    )
    model.summary()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    print("\nTraining model...")
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    print("\nSaving model...")
    model.save('sign_language_model.h5')
    print("Model saved successfully!")

    print("\nStarting real-time camera test...")
    print("Press 'q' to quit the camera window")
    test_with_camera(model, label_binarizer)
