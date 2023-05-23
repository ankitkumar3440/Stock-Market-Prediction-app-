import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


# start ='2010-01-01'
# end='2019-12-31'
st.title('Stock trend Prediction')

user_input=st.text_input("enter the Stock Ticker",'AAPL')
df=pd.read_csv('TSLA.csv')

st.subheader('Data from 2010-2019')
st.write(df.describe())

st.subheader('Closing Price vs Time chart wkith 100ma')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart wkith 100ma and 200ma')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close ,'b')
st.pyplot(fig)

data_trianing=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))]
    )
print(data_trianing.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_trianing_array=scaler.fit_transform(data_trianing)

# x_train=[]
# y_train=[]
#
# for i in range(100,data_trianing_array.shape[0]):
#     x_train.append(data_trianing_array[i-100:i])
#     y_train.append(data_trianing_array[i:0])
#
# x_train, y_train=np.array(x_train),np.array(y_train)

model=load_model('keras_model.h5')

pass_100_days=data_trianing.tail(100)

final_df=pass_100_days._append(data_testing, ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test=np.array(x_test),np.array(y_test)

y_predicted=model.predict(x_test)

scaler=scaler.scale_

scaler_factor=1/scaler[0]
y_predicted=y_predicted*scaler_factor
y_test=y_test*scaler_factor

# final Grpah
st.subheader('Predictions Vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)