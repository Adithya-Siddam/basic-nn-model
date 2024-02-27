# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
A Neural Network Regression Model leverages the power of interconnected artificial neurons to learn and predict continuous numerical values from input data, making it a valuable tool for tasks like forecasting trends in finance, healthcare, and scientific research.
The provided neural network schema depicts a simple, three-layered architecture commonly found in deep learning applications. It starts with an input layer containing one neuron, likely representing a single numerical value. This is followed by two hidden layers, each using the ReLU (Rectified Linear Unit) activation function. The first hidden layer has five neurons, while the second has four. These layers extract progressively complex features from the input data. Finally, the network concludes with a single output neuron, responsible for making a prediction or generating a final result. This basic structure allows the network to learn intricate relationships within the data and perform tasks like classification or regression.
```
Input layer: 1 neuron.
First hidden layer: 5 neurons with ReLU activation function.
Second hidden layer: 4 neurons with ReLU activation function.
Output layer: 1 neuron.
```
## Neural Network Model

![image](https://github.com/Adithya-Siddam/basic-nn-model/assets/93427248/c39fba98-2f48-4b77-92bb-f393f477fe67)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: S Adithya Chowdary.
### Register Number: 212221230100.
```python

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds,_ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('ex1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'float'})
df = df.astype({'output':'float'})

X = df[['input']].values
y = df[['output']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

AA1_model = Sequential([
    Dense(units = 5, activation = 'relu', input_shape=[1]),
    Dense(units = 4, activation = 'relu'),
    Dense(units = 1)
])

AA1_model.compile(optimizer= 'rmsprop', loss="mse")
AA1_model.fit(X_train1,y_train,epochs=5000)
AA1_model.summary()

loss_df = pd.DataFrame(AI_Brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
AA1_model.evaluate(X_test1,y_test)
X_n1 = [[20]]
X_n1_1 = Scaler.transform(X_n1)
AA1_model.predict(X_n1_1)


```
## Dataset Information

![image](https://github.com/Adithya-Siddam/basic-nn-model/assets/93427248/2f85dcbf-e9fa-4408-97cc-669ecbe28787)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/Adithya-Siddam/basic-nn-model/assets/93427248/f69b4414-94ab-46d7-b92c-cc024e2acfd7)

### Test Data Root Mean Squared Error

![image](https://github.com/Adithya-Siddam/basic-nn-model/assets/93427248/8c4c278a-6bab-4fa8-a348-523948242e49)

### New Sample Data Prediction

![image](https://github.com/Adithya-Siddam/basic-nn-model/assets/93427248/23427b58-7379-473e-a6db-c8e9b40ae910)

## RESULT

Thus, The Process of developing a neural network regression model for the created dataset is successfully executed.

