import pandas as pd
import numpy as np
import streamlit as st
import sklearn 
import imblearn
import pickle 
from PIL import Image

# load dataset
df = pd.read_csv('clean_dataset.csv')

# load model
model = pickle.load(open('xgb_fix_tuned.pkl','rb'))
    
# create title (homepage)
def main():
    load_image = Image.open('./image.jpg')
    st.image(load_image)
    st.title('The Student Adaptivity Level Prediction in Online Learning')
    st.subheader('Please input in the option box below!')

    # choose menu input - Selectbox
    # st.sidebar.subheader('Select Your Input!')
    gender = st.selectbox('Select Your Gender!', df['gender'].unique())
    age = st.selectbox('Select Your Age!', df['age'].unique())
    education_level = st.selectbox('Select Your Education Level!', df['education_level'].unique())
    institution_type = st.selectbox('Select Your Institution Type!', df['institution_type'].unique())
    it_student = st.selectbox('Are You an IT student?', df['it_student'].unique())
    location = st.selectbox('Is Your Location in The City/Town?', df['location'].unique())
    load_shedding = st.selectbox('Select Your Load Shedding!', df['load_shedding'].unique())
    financial_condition = st.selectbox('Select Your Financial Condition!', df['financial_condition'].unique())
    internet_type = st.selectbox('Select Your Internet Type!', df['internet_type'].unique())
    class_duration = st.selectbox('Select Your Class Duration!', df['class_duration'].unique())
    self_lms = st.selectbox("Does Your Institution Have it's Own LMS?", df['self_lms'].unique())
    device = st.selectbox('Select Your Device!', df['device'].unique())

    # prediction - button for predict
    if st.button('Predict'):
    # input the data in dataframe
        input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'education_level': [education_level],
        'institution_type': [institution_type],
        'it_student': [it_student],
        'location': [location],
        'load_shedding': [load_shedding],
        'financial_condition': [financial_condition],
        'internet_type': [internet_type],
        'class_duration': [class_duration],
        'self_lms': [self_lms],
        'device': [device],
        
        })

        # do predict with model
        prediction = model.predict(input_data)

        st.subheader('Prediction Result')
        if prediction[0] == 0:
            st.markdown('<div style="background-color: red; padding: 10px; border-radius: 5px;">The Adaptability Level of Student is Low.</div>', unsafe_allow_html=True)
        elif prediction[0] == 1:
            st.markdown('<div style="background-color: yellow; padding: 10px; border-radius: 5px;">The Adaptability Level of Student is Moderate.</div>', unsafe_allow_html=True)
        elif prediction[0] == 2:
            st.markdown('<div style="background-color: green; padding: 10px; border-radius: 5px;">The Adaptability Level of Student is High.</div>', unsafe_allow_html=True)
        else:
            st.success('<div style="background-color: grey; padding: 10px; border-radius: 5px;">Unknown Student Adaptability.</div>', unsafe_allow_html=True)

    st.write('----')
    st.write('''
    Dashboard Created by [Tyovendi Arisandy](https://www.linkedin.com/in/tyovendiarisandy/)
    ''')

if __name__=='__main__':
    main()
