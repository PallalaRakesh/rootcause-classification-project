# 📊 Root Cause Analysis with Deep Learning

This project demonstrates how to build a neural network to identify the root cause of an incident based on a set of system metrics. The model classifies issues into three categories: **MEMORY\_LEAK**, **NETWORK\_DELAY**, or **DATABASE\_ISSUE**.

## ⚙️ Project Setup

First, you need to set up your Python environment and install the required libraries. This project uses **TensorFlow**, **Keras**, and **scikit-learn** for building and training the model, and **pandas** and **NumPy** for data handling.

```bash
pip install tensorflow scikit-learn pandas numpy
```

## 📋 Dataset

The model is trained on a CSV file named `root_cause_analysis.csv`. This dataset contains the following columns:

  * **`ID`**: A unique identifier for each incident.
  * **`CPU_LOAD`**, **`MEMORY_LEAK_LOAD`**, **`DELAY`**: Numerical metrics representing the state of the system at the time of the incident. These are binary flags (0 or 1).
  * **`ERROR_1000`**, **`ERROR_1001`**, **`ERROR_1002`**, **`ERROR_1003`**: Flags indicating which specific errors were logged. These are also binary (0 or 1).
  * **`ROOT_CAUSE`**: The target label, which is the actual cause of the incident.

## 📝 Step-by-Step Explanation

### 1\. Data Preprocessing

Data preprocessing is a crucial step to prepare the raw data for the neural network.

  * **Loading Data**: The `pandas` library is used to load the CSV file into a DataFrame.
  * **Label Encoding**: The `ROOT_CAUSE` column, which contains categorical text values (`MEMORY_LEAK`, `NETWORK_DELAY`, etc.), is converted into numerical integers using `LabelEncoder`. This is a necessary step because machine learning models only work with numbers.
  * **One-Hot Encoding**: The integer labels are then converted into a **one-hot encoded vector**. This means each category becomes a separate column with a binary value (1 or 0) indicating its presence. For example, `MEMORY_LEAK` might be represented as `[1, 0, 0]`, `NETWORK_DELAY` as `[0, 1, 0]`, and `DATABASE_ISSUE` as `[0, 0, 1]`. This is the standard format for multi-class classification problems with a softmax output layer.
  * **Splitting Data**: The data is split into **training (90%)** and **testing (10%)** sets. The model learns from the training data and is then evaluated on the unseen test data to ensure it can generalize to new incidents.

### 2\. Model Architecture

The model is a simple **sequential neural network** built with Keras. It has a classic structure with two hidden layers.

  * **Input Layer**: A `Dense` (fully connected) layer that takes in the 7 symptom flags (`CPU_LOAD` through `ERROR_1003`) as input. It has 128 neurons and uses the **ReLU (Rectified Linear Unit)** activation function, which is efficient and helps the model learn complex, non-linear patterns.
  * **Hidden Layer**: A second `Dense` layer with 128 neurons and a ReLU activation function. This layer deepens the network and allows it to capture more complex relationships between the input features.
  * **Output Layer**: The final `Dense` layer has 3 neurons, one for each of the three root cause classes. It uses a **softmax** activation function. Softmax converts the raw output of the neurons into a probability distribution, ensuring the sum of the probabilities for all classes is 1. The class with the highest probability is the model's final prediction.

### 3\. Training and Evaluation

The model is compiled with an optimizer and loss function, and then trained to find the best set of internal parameters.

  * **Loss Function**: **`categorical_crossentropy`** is used to measure the difference between the model's predicted probabilities and the true one-hot encoded labels. The goal of the training process is to minimize this loss.
  * **Training**: The model is trained over **20 epochs** using batches of 64 samples. It's validated on a portion of the training data to monitor for overfitting.
  * **Evaluation**: The trained model is evaluated on the separate test dataset, providing a final accuracy score. In the provided notebook, the model achieves an accuracy of **87%** on the test data.

### 4\. Making Predictions

Once the model is trained, you can use it to predict the root cause of new incidents. To do this, you simply pass the 7 system metrics (as a list or a batch of lists) to the model. The model will output a probability vector, and `np.argmax` is used to find the index of the highest probability, which corresponds to the predicted class.

## 🚀 How to Run the Project

1.  Clone this repository to your local machine.
2.  Install the required libraries.
3.  Place your `root_cause_analysis.csv` file in the project directory.
4.  Open and run the `code-rootcause-analysis.ipynb` Jupyter Notebook to execute the code.
