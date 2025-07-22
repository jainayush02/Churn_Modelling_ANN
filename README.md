This project is an implementation of an Artificial Neural Network (ANN) to predict customer churn for a banking institution. The model is built using TensorFlow and Keras on the widely-used "Churn_Modelling.csv" dataset. The primary objective is to identify customers who are likely to leave the bank, which can enable the bank to take proactive measures to retain them.

Dataset
The project utilizes the Churn_Modelling.csv dataset, which contains various customer attributes that may influence their decision to leave the bank. The key features used for prediction include:

CreditScore: The credit score of the customer.

Geography: The country where the customer resides (France, Spain, Germany).

Gender: The gender of the customer.

Age: The age of the customer.

Tenure: The number of years the customer has been with the bank.

Balance: The account balance of the customer.

NumOfProducts: The number of products the customer has with the bank.

HasCrCard: Whether the customer has a credit card (1 for yes, 0 for no).

IsActiveMember: Whether the customer is an active member (1 for yes, 0 for no).

EstimatedSalary: The estimated salary of the customer.

The target variable is Exited, which indicates whether a customer has churned (1 for yes, 0 for no).

Methodology
The project follows a standard machine learning workflow:

Data Preprocessing:

Categorical features like Geography and Gender are converted into numerical format using one-hot encoding.

The dataset is split into training and testing sets, with 80% of the data used for training and 20% for testing.

Feature scaling is applied using StandardScaler to normalize the range of the independent variables.

Model Architecture:

An ANN is constructed using the Keras Sequential API.

The model consists of an input layer, two hidden layers with the ReLU activation function, and an output layer with the Sigmoid activation function, suitable for binary classification.

The model is compiled with the Adam optimizer and uses binary cross-entropy as the loss function.

Training and Evaluation:

The model is trained for 100 epochs with a batch size of 10.

An EarlyStopping callback is used to prevent overfitting by monitoring the validation loss.

The model's performance is evaluated on the test set using a confusion matrix and accuracy score.

Results
The trained model achieves an accuracy of approximately 86.15% on the test data. The confusion matrix provides a more detailed breakdown of the model's performance:

True Positives: 205

True Negatives: 1518

False Positives: 77

False Negatives: 200

These results indicate that the model is effective at predicting customer churn, providing a valuable tool for customer retention strategies.

How to Run the Code
To replicate this project, you will need a Python environment with the following libraries installed:

pandas

NumPy

scikit-learn

TensorFlow

You can run the provided Jupyter Notebook (Churn_Modelling_ANN.ipynb) in an environment like Jupyter or Google Colab. The notebook is self-contained and includes all the necessary code to load the data, preprocess it, build the model, train it, and evaluate its performance. Ensure that the Churn_Modelling.csv dataset is in the same directory as the notebook.
