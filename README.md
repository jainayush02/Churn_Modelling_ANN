🏦 Customer Churn Prediction with an ANN 🤖

A project focused on predicting customer churn in a bank using an Artificial Neural Network (ANN). This model is built with TensorFlow and Keras to help identify customers who are likely to close their accounts.

📊 Dataset

This project uses the Churn_Modelling.csv dataset, which contains key customer information. The features used for prediction are:

CreditScore 💳

Geography 🌍 (France, Spain, Germany)

Gender 🚻

Age 🧑

Tenure 📅

Balance 💰

NumOfProducts 🛍️

HasCrCard 💳

IsActiveMember 🏃

EstimatedSalary 💵

The target variable is Exited, indicating whether a customer has churned (1 for yes, 0 for no).

🛠️ Methodology
The project follows these steps:

Data Preprocessing 🧹:

Categorical features (Geography, Gender) are one-hot encoded.

The dataset is split into training (80%) and testing (20%) sets.

StandardScaler is used for feature scaling to normalize the data.

Model Architecture 🧠:

A Sequential model is built using the Keras API.

The architecture includes an input layer, two hidden layers with ReLU activation, and a Sigmoid activation function in the output layer for binary classification.

The model is compiled with the Adam optimizer and uses binary cross-entropy as the loss function.

Training and Evaluation 📈:

The model is trained for up to 100 epochs with a batch size of 10.

EarlyStopping is implemented to prevent overfitting.

The model's performance is evaluated using a confusion matrix and accuracy score.

🚀 Results
The model achieves an accuracy of approximately 86.15% on the test data! 🎉

Here's a look at the confusion matrix:

Predicted Negative	Predicted Positive
Actual Negative	1518	77
Actual Positive	200	205

Export to Sheets
These results show that the model is quite effective at identifying customers who are likely to churn.

💻 How to Run the Code
To run this project on your own, you'll need a Python environment with these libraries:

pandas

numpy

scikit-learn

tensorflow

You can run the Churn_Modelling_ANN.ipynb notebook in an environment like Jupyter or Google Colab. Make sure the Churn_Modelling.csv dataset is in the same directory.
