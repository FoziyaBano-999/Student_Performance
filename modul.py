import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler  

df = pd.read_csv('Student_Performance.csv')

# converting categoricale data into numerical
df['Extracurricular Activities'] = df['Extracurricular Activities'].apply(lambda x : 1 if x == "Yes" else 0)

#  split input and output
x = df.drop('Performance Index' , axis=1)
y = df['Performance Index']

# divide into 75% and 25%
x_train, x_test, y_train, y_test = train_test_split(x,y , test_size=0.25 , random_state=42) 

# scalling our splited data
scaller = StandardScaler()

scaled_train = scaller.fit_transform(x_train)
x_train[x_train.columns] = scaled_train

# now train our modul
modul = LinearRegression()
modul.fit(x_train , y_train)

# now predict our value 

y_predict = modul.predict(x_test)
# st.write(f'{y_predict}')
# [54.73187888 22.61211054 47.90838844 ... 68.07396952 53.68636805 54.85816372]
rmse = root_mean_squared_error(y_test , y_predict)
# st.write(f'{rmse}')
# 2.008119571992444

st.title('🏫 Student performance prediction web app')

hour = st.number_input('Enter the number of hourse you studies' , min_value=1 , max_value= 20)
previous_score = st.number_input('Enter your previous score' ,min_value=0 , max_value= 100)
sleep_hours = st.number_input('Enter yor sleep hours'  ,min_value=1 , max_value= 15 )
sample_paper = st.number_input('Number of sample paper you solved'  ,min_value=1 , max_value= 30)
extra_activity = st.radio('Extra curricular activity' ,['Yes','No'])

activity = 1 if extra_activity == "Yes" else 0

if st.button('Predict performance value'):
    user_data =[[hour , previous_score ,sleep_hours,sample_paper,activity]]
    scalled_user_data = scaller.transform(user_data)
    prediction = modul.predict(scalled_user_data)
    round_pred = round(prediction[0] ,2)
    st.success(f'your performance is {round_pred}%')


