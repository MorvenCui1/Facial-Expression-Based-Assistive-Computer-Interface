import sqlite3

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import os
#Path to files
base_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base_dir, "eyeDatabase.db")
data_path = os.path.join(base_dir, "eyeCloseData.txt")

#Connect to database or create it if it doesn't exist
conn = sqlite3.connect(db_path)

#Create table if table does not exist
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS eyeData (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    eyegap REAL NOT NULL,
    eyeclose INTEGER NOT NULL
)
''')

#Delete all rows from table
#cursor.execute("DELETE FROM eyeData")

#Read data in from file to insert into table
data = []
with open(data_path, "r") as file:
    for line in file:
        gap, close = line.strip().split(",")
        data.append((float(gap), int(close)))

#Insert all data into table
cursor.execute("SELECT COUNT(*) FROM eyeData")
count = cursor.fetchone()[0]
if count == 0:
    cursor.executemany('''
        INSERT INTO eyeData (eyegap, eyeclose)
        values (?, ?)
        ''', [(gap, int(close)) for gap, close in data])

#Insert all data from database into Pandas dataframe
df = pd.read_sql_query("SELECT * FROM eyeData", conn)

conn.commit() #Commit changes to database
conn.close() #Closes the connection to database

X = df[['eyegap']] #2D array-like features
y = df['eyeclose'] #Target variable

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.1, random_state = 42
)

#Machine learning models
#------------------------------------------------------------------------------------
#Create logistic regression model with training data
model1 = LogisticRegression()
model1.fit(X_train, y_train)
print("Logistic regression model")

#Predict on test set
y_pred = model1.predict(X_test)
y_pred_class = (y_pred >= 0.5).astype(int) #Converts floats to booleans

#Evaluate performance of model
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report: \n", classification_report(y_test, y_pred))

#------------------------------------------------------------------------------------
#Create linear regression model with training data
model2 = LinearRegression()
model2.fit(X_train, y_train)
print("Linear regression model")

#Predict on test set
y_pred = model2.predict(X_test)
y_pred_class = (y_pred >= 0.5).astype(int) #Converts floats to booleans

#Evaluate performance of model
print("Accuracy: ", accuracy_score(y_test, y_pred_class))
print("\nClassification Report: \n", classification_report(y_test, y_pred_class))

#------------------------------------------------------------------------------------
#Create k neighbors model with training data
model3 = KNeighborsClassifier(n_neighbors=20)
model3.fit(X_train, y_train)
print("k Neighbors model")

#5-fold cross validation, train on 4 parts and test on 1
scores = cross_val_score(model3, X, y, cv = 5, scoring = 'accuracy')

#Print cross validation backtesting results
print("Cross-validation accuracies: ", scores)
print("Mean accuracy: ", scores.mean())

#Predict on test set
y_pred = model3.predict(X_test)

#Evaluate performance of model
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report: \n", classification_report(y_test, y_pred))

#------------------------------------------------------------------------------------
#Create decision tree model with training data
model4 = DecisionTreeClassifier(random_state=20)
model4.fit(X_train, y_train)
print("Decision Tree model")

#5-fold cross validation, train on 4 parts and test on 1
scores = cross_val_score(model4, X, y, cv = 5, scoring = 'accuracy')

#Print cross validation backtesting results
print("Cross-validation accuracies: ", scores)
print("Mean accuracy: ", scores.mean())

#Predict on test set
y_pred = model4.predict(X_test)

#Evaluate performance of model
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report: \n", classification_report(y_test, y_pred))

#------------------------------------------------------------------------------------
#Create random forest model with training data
model5 = RandomForestClassifier(n_estimators = 75, random_state = 42)
model5.fit(X_train, y_train)
print("Random forest model")

#5-fold cross validation, train on 4 parts and test on 1
scores = cross_val_score(model5, X, y, cv = 5, scoring = 'accuracy')

#Print cross validation backtesting results
print("Cross-validation accuracies: ", scores)
print("Mean accuracy: ", scores.mean())

# Predict on test set and evaluate accuracy
y_pred = model5.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("Test cases: ")
models = [model1, model2, model3, model4, model5]
for i in range (len(models)):
    model = models[i]
    print("----------------------------")
    print(f"Model {i+1}: {type(model).__name__}")

    #Predict for new samples
    new_eyegap = [[0.010]]
    prediction1 = model.predict(new_eyegap)
    print("Eyegap: 0.010")
    print("Predicted eyeclose: ", prediction1[0])
    print("Expected prediction: 1")

    new_eyegap = [[0.020]]
    prediction2 = model.predict(new_eyegap)
    print("Eyegap: 0.020")
    print("Predicted eyeclose: ", prediction2[0])
    print("Expected prediction: 0")

    new_eyegap = [[0.015]]
    prediction3 = model.predict(new_eyegap)
    print("Eyegap: 0.015")
    print("Predicted eyeclose: ", prediction3[0])
    print("Expected prediction: 1")

    new_eyegap = [[0.016]]
    prediction4 = model.predict(new_eyegap)
    print("Eyegap: 0.016")
    print("Predicted eyeclose: ", prediction4[0])
    print("Expected prediction: 1")

    new_eyegap = [[0.018]]
    prediction5 = model.predict(new_eyegap)
    print("Eyegap: 0.018")
    print("Predicted eyeclose: ", prediction5[0])
    print("Expected prediction: 0")