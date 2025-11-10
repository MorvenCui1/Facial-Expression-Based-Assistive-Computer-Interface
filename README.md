# Facial-Expression-Based-Assistive-Computer-Interface
Uses facial landmarks of user as inputs to replace a keyboard and mouse, designed for users with impaired motor functions. Uses movement of nose to control cursor, left eye winking to left click, right eye winking to right click, and open mouth to bring up and down keyboard.  
Uses MediaPipe and OpenCV to capture facial positions as inputs, uses PyAutoGUI to automate mouse and keyboard functions based on detected motions. Stores user facial landmarks data on SQLite database for future use in training machine learning models. Used scikit-learn to adapt to facial landmarks of each user by testing machine learning models such as linear regression, logistic regression, decision trees, k nearest neighbors, and random forest. Stores trained machine learning models using joblib.  

Used libraries: sqlite3, pandas, sklearn, joblib, cv2, mediapipe, pyautogui, timeit, os
