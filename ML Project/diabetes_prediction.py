#loading the data
import pandas as pd
diabetes_data = pd.read_csv('diabetes_prediction_dataset.csv')


#to remove the duplicates rows from the dataset
diabetes_data.drop_duplicates(inplace=True)
diabetes_data
#diabetes_data['gender'] = diabetes_data['gender'].replace({'Male': 1, 'Female': 0})
#print(diabetes_data)


diabetes_data = pd.get_dummies(diabetes_data, columns=['gender'], drop_first=True)
diabetes_data = pd.get_dummies(diabetes_data, columns=['smoking_history'], drop_first=True)
print(diabetes_data)



#split the data into test and train 
from sklearn.model_selection import train_test_split
X = diabetes_data.drop('diabetes', axis=1)
Y = diabetes_data['diabetes']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


#to train the model 
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()
Y_train = Y_train.values.reshape(-1, 1)
model_rf.fit(X_train, Y_train)
Y_test_reshaped = Y_test.values.reshape(-1, 1)


#to test the model and find the accuracy of the model 
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    Y_pred_binary = [0 if pred < 0.5 else 1 for pred in Y_pred]
    accuracy = accuracy_score(Y_test, Y_pred_binary)
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    return mse, r2, accuracy
mse_rf, r2_rf, accuracy_rf = evaluate_model(model_rf, X_test, Y_test)
print("Model 2 - Random Forest")
print("Mean Squared Error:", mse_rf)
print("R-squared:", r2_rf)
print("Accuracy: ", accuracy_rf)


#user_interface
def predict_diabetes(user_input):
    # Create a DataFrame from user input
    user_df = pd.DataFrame(user_input, index=[0])

    user_df = user_df[X_train.columns]

    # Make the prediction using the trained model
    user_prediction = model_rf.predict(user_df)

    if user_prediction[0] == 0:
        return "The model predicts that the user does not have diabetes."
    else:
        return "The model predicts that the user has diabetes."

# Example user input (replace with actual user input)
user_input = {
    'gender_Male': int(input("Enter 0 or 1 for female and male: ")),
    'age': int(input("Enter age: ")),
    'hypertension': int(input("Enter 0 for false and 1 for true if you have hypertension: ")),
    'heart_disease': int(input("Enter 0 for false and 1 for true if you have heart disease: ")),
    'smoking_history_never': int(input("Enter 0 for false and 1 for true if you have smoking history: ")),
    'bmi': float(input("Enter your BMI value: ")),
    'HbA1c_level': float(input("Enter your HbA1c level: ")),
    'blood_glucose_level': int(input("Enter your blood glucose level: ")),
    'gender_Other': 0,                   # Set these values as 0 for now
    'smoking_history_current': 0,        # Set these values as 0 for now
    'smoking_history_ever': 0,           # Set these values as 0 for now
    'smoking_history_former': 0,         # Set these values as 0 for now
    'smoking_history_not current': 0     # Set these values as 0 for now
}

# Run the prediction function with user input
result = predict_diabetes(user_input)
print(result)


#to serialize the data when into any website
import pickle 
with open('model_rf.pkl', 'wb') as model_file:
    pickle.dump(model_rf, model_file)
    
with open('model_rf.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
