
#import các thư viện liên quan đến Firebase
import firebase_admin
from firebase_admin import credentials, db
import ast
import time
#Certificate của Firebase
cred = credentials.Certificate("C:/Users/DELL/Desktop/doan_lastfile/training-ec468-firebase-adminsdk-n8d95-a47fea2f84.json")
firebase_admin.initialize_app(cred, {'databaseURL': 'https://training-ec468-default-rtdb.asia-southeast1.firebasedatabase.app'})


#import các thư viện liên quan đến model
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from scipy import signal
import sklearn.metrics
from sklearn.metrics import confusion_matrix
import seaborn as sb



#Function Build CNN (Mạng Noron tích chập)
def build_cnn(input_shape,number_of_activities,seq=0): 
    input_layer = keras.Input(shape = input_shape,name = 'title_'+str(seq))
    
    #lop Cov1D voi 24 bo loc, kich thuoc bo loc 2, buoc nhay 1 va phu bang same duoc ap dung len dau vao
    cnn = layers.Conv1D(24,2,1,"same",name = 'Conv1D_'+str(seq)+'_1')(input_layer)
    #LayerNormalization duoc ap dung len ket qua cua Con1vD truoc do
    cnn = layers.LayerNormalization(name = 'layernorm_'+str(seq)+'_1')(cnn)
    #Dropout voi ti le 0.1 duoc ap dung ngau nhien loai bo phan tu cua dau ra tu lop trc do
    cnn = layers.Dropout(rate = 0.1)(cnn)
    
    #Conv1D khác với 144 bộ lọc, kích thước bộ lọc 2, bước nhảy 1 và phủ bằng "same".
    cnn = layers.Conv1D(144,2,1,"same",name = 'Conv1D_2'+str(seq)+'_1')(cnn)
    #Lớp LayerNormalization và lớp MaxPool1D được áp dụng lần lượt sau đó.
    cnn = layers.LayerNormalization(name = 'layernorm_2'+str(seq)+'_1')(cnn)
    cnn = layers.MaxPool1D(2)(cnn)

    cnn = layers.Conv1D(288,2,1,"same",name = 'Conv1D_3'+str(seq)+'_1')(cnn)
    cnn = layers.LayerNormalization(name = 'layernorm_3'+str(seq)+'_1')(cnn)
    cnn = layers.MaxPool1D(2)(cnn)
    cnn = layers.Dropout(rate = 0.1)(cnn)
    
    cnn = layers.Conv1D(512,2,1,"same",name = 'Conv1D_4'+str(seq)+'_1')(cnn)
    cnn = layers.LayerNormalization(name = 'layernorm_4'+str(seq)+'_1')(cnn)
    cnn = layers.MaxPool1D(2)(cnn)
    
    cnn = layers.Conv1D(288,2,1,"same",name = 'Conv1D_5'+str(seq)+'_1')(cnn)
    cnn = layers.LayerNormalization(name = 'layernorm_5'+str(seq)+'_1')(cnn)
    cnn = layers.MaxPool1D(2)(cnn)
    cnn = layers.Dropout(rate = 0.1)(cnn)
    
    cnn = layers.Conv1D(144,2,1,"same",name = 'Conv1D_6'+str(seq)+'_1')(cnn)
    cnn = layers.LayerNormalization(name = 'layernorm_6'+str(seq)+'_1')(cnn)
    cnn = layers.MaxPool1D(2)(cnn)
    #Một lớp Flatten được sử dụng để biến đổi đầu ra từ các lớp Conv1D cuối cùng thành một vector 1 chiều.    
    cnn = layers.Flatten()(cnn)
    #Các lớp Dense với các hàm kích hoạt relu được sử dụng để ánh xạ các đặc trưng 
    #đã được trích xuất từ lớp trước thành các đặc trưng tương ứng với số lượng nơ-ron được chỉ định.
    cnn = layers.Dense(144,activation='relu',name = 'dense_1')(cnn)
    cnn = layers.Dense(72,activation='relu',name = 'dense_2')(cnn)
    cnn = layers.Dense(24,activation='relu',name = 'dense_3')(cnn)
    #Lớp Dense cuối cùng có số lượng nơ-ron bằng number_of_activities 
    #và hàm kích hoạt softmax để tính toán xác suất của từng lớp đầu ra.
    cnn = layers.Dense(number_of_activities,activation='softmax',name = 'dense_4')(cnn)
    #Cuối cùng, một đối tượng Model được tạo với đầu vào là lớp Input và đầu ra là đầu ra của lớp Dense cuối cùng. Mô hình được đặt tên là 'model'.
    return keras.Model(input_layer,cnn,name = 'model')



#Lấy dữ liệu
subjects = np.arange(1,40)
tasks = np.arange(20,35)
trials = np.arange(1,12)

xtrain = []
ytrain = []

for i in subjects:
    time1 = time.time()
    print('Reading subject ',i,end=' , ')
    for j in tasks:
        for k in trials:
            try:

                filename = 'D'+ str(i).zfill(2) + 'T' + str(j).zfill(2) + 'R' + str(k).zfill(2) + '.csv'
                labelfilename =  'D'+ str(i).zfill(2) + '_label.xlsx'

                label = pd.read_excel('C:/Users/DELL/Desktop/doan_lastfile/data/label_data'+labelfilename,index_col=None, header=None)[1:]
                data = pd.read_csv('C:/Users/DELL/Desktop/doan_lastfile/data/sensor_data'+'D' + str(i).zfill(2)+'/'+filename)

                acc_x = np.array(data['AccX']).reshape((len(data),1))
                acc_y = np.array(data['AccY']).reshape((len(data),1))
                acc_z = np.array(data['AccZ']).reshape((len(data),1))
                gyro_x = np.array(data['GyrX']).reshape((len(data),1))
                gyro_y = np.array(data['GyrY']).reshape((len(data),1))
                gyro_z = np.array(data['GyrZ']).reshape((len(data),1))
                euler_x = np.array(data['EulerX']).reshape((len(data),1))
                euler_y = np.array(data['EulerY']).reshape((len(data),1))
                euler_z = np.array(data['EulerZ']).reshape((len(data),1))

                data = np.concatenate([acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,euler_x,euler_y,euler_z],axis = -1)

                activity = 0
                for row in range(len(label)):
                    trial = int(label.iloc[row][2])
                    if not pd.isnull(label.iloc[row][0]):
                        activity = int(label.iloc[row][0][5:7])
                    start = label.iloc[row][3]-1
                    end = label.iloc[row][4]-1
                    
                    if end - start < 50 : continue
                    
                    if trial == k and activity == j :
                        x = int(start+((end - start) - 50 )/2)
                        xtrain.append(data[x:x+50])
                        ytrain.append(1)
                        
            except : continue
    time2 = time.time()
    print('Time taken = ',time2-time1)


for i in subjects:
    time1 = time.time()
    print('Reading subject ',i,end=' , ')
    for j in np.arange(1,40):
        if j in tasks : continue
        for k in trials:
            try :
                filename = 'S'+ str(i).zfill(2) + 'T' + str(j).zfill(2) + 'R' + str(k).zfill(2) + '.csv'
                
                data = pd.read_csv('C:/Users/DELL/Desktop/doan_lastfile/data/sensor_data/'+'D' + str(i).zfill(2)+'/'+filename)

                acc_x = np.array(data['AccX']).reshape((len(data),1))
                acc_y = np.array(data['AccY']).reshape((len(data),1))
                acc_z = np.array(data['AccZ']).reshape((len(data),1))
                gyro_x = np.array(data['GyrX']).reshape((len(data),1))
                gyro_y = np.array(data['GyrY']).reshape((len(data),1))
                gyro_z = np.array(data['GyrZ']).reshape((len(data),1))
                euler_x = np.array(data['EulerX']).reshape((len(data),1))
                euler_y = np.array(data['EulerY']).reshape((len(data),1))
                euler_z = np.array(data['EulerZ']).reshape((len(data),1))
               

                data = np.concatenate([acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,euler_x,euler_y,euler_z],axis = -1)

                cnt = 0
                if data.shape[0] < 100 : continue
                
                if j in [1,11,12,17] : 
                    for x in range(50,data.shape[0],50):
                        xtrain.append(data[x:x+50])
                        ytrain.append(0)
                        cnt += 1
                        if cnt == 10:break
                else : 
                    for x in range(50,data.shape[0],50):
                        xtrain.append(data[x:x+50])
                        ytrain.append(0)
                        cnt += 1
                        if cnt == 2:break
                
            except : continue
    time2 = time.time()
    print('Time taken = ',time2-time1)

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)


#chuyển ytrain về dạng vector 2 chiều
ytrain1 = []
for i in ytrain:
    ytrain1.append([0]*2)
    ytrain1[-1][i] = 1
ytrain = np.array(ytrain1)

#chia xtrain, ytrain chiem 70%; xtest, ytest, xval, yval moi cai 15%
xtrain,xtest,ytrain,ytest = train_test_split(xtrain,ytrain,train_size = 0.7)
xtest,xval,ytest,yval = train_test_split(xtest,ytest,train_size = 0.5)


#Chuan hoa cac min-max
#Bản chất là đừa giá trị của tín hiệu về đọna -1 đến 1 => giúp đồn nhất phạm vi giá trị và tạo ra một độ phân bổ cân đối hơn cho các tín hiệu
for i in range(9): #duyet tu 0->9
    min_ = min([min(j) for j in xtrain[:,:,i]])
    max_ = max([max(j) for j in xtrain[:,:,i]])
    
    xtrain[:,:,i] = 2*(xtrain[:,:,i]-min_)/(max_-min_)-1
    
for i in range(9):
    min_ = min([min(j) for j in xtest[:,:,i]])
    max_ = max([max(j) for j in xtest[:,:,i]])
    
    xtest[:,:,i] = 2*(xtest[:,:,i]-min_)/(max_-min_)-1
    
for i in range(9):
    min_ = min([min(j) for j in xval[:,:,i]])
    max_ = max([max(j) for j in xval[:,:,i]])
    
    xval[:,:,i] = 2*(xval[:,:,i]-min_)/(max_-min_)-1

cnn = build_cnn(xtrain.shape[1:],2)
cnn.summary()



# Training cnn
cnn.compile(loss = 'categorical_crossentropy',optimizer='AdaGrad',metrics=['accuracy'])
cnn_history = cnn.fit(
    xtrain,
    ytrain,
    validation_data = (
        xval,
        yval
    ),
    epochs = 30,
    batch_size = 200,
)

def get_data_from_firebase(i):
    while True:
        ref = db.reference(f'input_data/data{i}')
        data_string = ref.get()
        #print(data_string)
        # Kiểm tra xem có dữ liệu mới (data{i}) trong Firebase hay không
        if data_string is not None:
            data_list = ast.literal_eval(data_string)
            data_list = eval(data_string)
            data_array = np.array(data_list)
            data_list = data_array.reshape(1, 50, 9)
            return data_list
        time.sleep(1)
i = 1       
while True:
    # Lấy dữ liệu từ Firebase
    data = get_data_from_firebase(i)
    #print("Dữ liệu từ Firebase:", data)
    predicted_probabilities = cnn.predict(data)
    prediction_class_0 = predicted_probabilities[0][0]
    prediction_class_1 = predicted_probabilities[0][1]
    print(f'data{i}:')
    print(predicted_probabilities)
    # Kiểm tra điều kiện và in ra kết quả
    if prediction_class_0 > prediction_class_1:
        print("Không ngã")
    else:
        print("Ngã")
        
    i += 1
    # Tạm dừng vòng lặp trong một khoảng thời gian (ví dụ: 5 giây) trước khi kiểm tra lại
    time.sleep(5)





