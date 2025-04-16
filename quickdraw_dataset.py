import os
import numpy as np
from PIL import Image
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from quickdraw import QuickDrawDataGroup

# quickdraw directory
def make_data_dirs(*dir_names: str, base_dir: str = "."):
    for dir_name in dir_names:
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

# quickdraw dataset
def gen_class_images(class_names: list, download_cache_dir: str, image_dir: str,
                     stroke_widths: list, max_drawings: int = 1000,
                     drawing_size=(28, 28), recognized: bool = True,
                     img_ext: str = "png"):
    for class_name in class_names:
        qdg = QuickDrawDataGroup(class_name, max_drawings=max_drawings,
                                 recognized=recognized, cache_dir=download_cache_dir)

        for i, drawing in enumerate(qdg.drawings):
            for width in stroke_widths:
                class_image_dir = f"{image_dir}/{class_name}"
                make_data_dirs(class_image_dir)
                filepath = f"{class_image_dir}/{class_name}_{i}_width_{width}.{img_ext}"
                drawing.get_image(stroke_width=width).resize(drawing_size).save(filepath)

# main neuralnetwork
class QuickDrawNN:
    def __init__(self, input_size=28 * 28, hidden1=512, hidden2=256, hidden3=128, num_classes=5, learning_rate=0.001):
        self.learning_rate = learning_rate

        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, hidden3) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros((1, hidden3))
        self.W4 = np.random.randn(hidden3, num_classes) * np.sqrt(2.0 / hidden3)
        self.b4 = np.zeros((1, num_classes))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.relu(self.Z3)
        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.A4 = self.softmax(self.Z4)
        return self.A4

    def compute_loss(self, Y_pred, Y_true):
        m = Y_true.shape[0]
        log_probs = -np.log(Y_pred[range(m), Y_true])
        return np.sum(log_probs) / m

    def backward(self, X, Y_true, Y_pred):
        # backprop
        m = X.shape[0]

        # one hot code
        Y_true_one_hot = np.zeros_like(Y_pred)
        Y_true_one_hot[np.arange(m), Y_true] = 1

        dZ4 = Y_pred - Y_true_one_hot
        dW4 = np.dot(self.A3.T, dZ4) / m
        db4 = np.sum(dZ4, axis=0, keepdims=True) / m

        dZ3 = np.dot(dZ4, self.W4.T) * self.relu_derivative(self.Z3)
        dW3 = np.dot(self.A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dZ2 = np.dot(dZ3, self.W3.T) * self.relu_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dZ1 = np.dot(dZ2, self.W2.T) * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W4 -= self.learning_rate * dW4
        self.b4 -= self.learning_rate * db4

# save weights
def save_model(model, weights_path: str = "model.npz"):
    weights = {
        'W1': model.W1, 'b1': model.b1,
        'W2': model.W2, 'b2': model.b2,
        'W3': model.W3, 'b3': model.b3,
        'W4': model.W4, 'b4': model.b4
    }
    np.savez(weights_path, **weights)
    print(f"Model weights saved to '{weights_path}'.")

# load weights
def load_model(weights_path: str = "model.npz"):
    model = QuickDrawNN(input_size=28 * 28, hidden1=512, hidden2=256, hidden3=128, num_classes=5)
    if os.path.exists(weights_path):
        weights = np.load(weights_path, allow_pickle=True)
        model.W1, model.b1 = weights['W1'], weights['b1']
        model.W2, model.b2 = weights['W2'], weights['b2']
        model.W3, model.b3 = weights['W3'], weights['b3']
        model.W4, model.b4 = weights['W4'], weights['b4']
        print(f"Model loaded from '{weights_path}'.")
    else:
        print(f"No saved model found at '{weights_path}'. Starting from scratch.")
    return model

# load quickdraw
def load_quickdraw_dataset(image_dir: str, classes: List[str], test_size=0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, Y = [], []
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(image_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = Image.open(img_path).convert('L').resize((28, 28))  # Convert to grayscale and resize
            X.append(np.array(img).flatten())
            Y.append(class_idx)

    X = np.array(X) / 255.0  # Normalize to [0, 1]
    Y = np.array(Y)
    return train_test_split(X, Y, test_size=test_size, random_state=42)

# train ##### CHANGE EPOCHS HERE #####
def train_model(model, X_train, Y_train, X_val, Y_val, epochs=100, batch_size=64): # increase epochs if you wish
    for epoch in range(epochs):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            Y_pred = model.forward(X_batch)
            loss = model.compute_loss(Y_pred, Y_batch)
            model.backward(X_batch, Y_batch, Y_pred)

        val_preds = model.forward(X_val)
        val_accuracy = np.mean(np.argmax(val_preds, axis=1) == Y_val)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

# main
def main():
    cache_dir = "./.quickdrawcache"
    image_dir = "./data/quickdraw"
    make_data_dirs(cache_dir, image_dir)

    ###### CHANGE DOODLES HERE ######
    image_types = ['bee', 'car', 'cat', 'dog', 'sailboat']
    stroke_widths = [2, 3]
    gen_class_images(image_types, cache_dir, image_dir, stroke_widths)

    X_train, X_val, Y_train, Y_val = load_quickdraw_dataset(image_dir, image_types)

    model = load_model()
    train_model(model, X_train, Y_train, X_val, Y_val)
    save_model(model)

if __name__ == "__main__":
    main()
