from msvcrt import kbhit
from tabnanny import verbose
import pandas as pd
import numpy as np
import keras
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import time
from keras.layers import LSTM
from keras.layers import Dense
from typing import List
from keras import backend as K, Model, Input, optimizers
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tcn import TCN, tcn_full_summary
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, GRU, Dropout, Conv1D, BatchNormalization, ReLU, Add, Flatten, concatenate ,multiply
from keras.models import Model, load_model
import random
from keras.utils import plot_model
from silence_tensorflow import silence_tensorflow
import data_load
import os

silence_tensorflow()
with tf.device('/GPU:1'):

    
    # 정규화 함수 정의
    def normalize(df, global_min, global_max):
        return (df - global_min) / (global_max - global_min)
    
    # 비정규화 함수 정의
    def denormalize(norm_df, global_min, global_max):
        return (norm_df * (global_max - global_min)) + global_min

    # trainm test 데이터셋 & 엣지서버 기준값 불러오기
    data_path = 'C:\\Users\\nayeon\\Desktop\\research\\edge_server\\datasets'
    data = pd.read_csv(data_path + '\\hourlyGridActivity.csv')

    # 전처리
    data = data_load.redifine_types_to_mb(data)

    # keras 모델 불러오기
    file_path = '3DATA_0208/3DATA_ALL/internet_0710_1hour_all_cells_new_data_step_3'
    predictions = []
    y_test_list = []

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for j in range(2):
        if j == 0  :
            load_data = pd.read_csv('900_Bus_240703_1h_total.csv') #bu_0213.csv
            model_path = file_path + '__' + str(j) + "_0710_rctl_model.h5"
            load_model = keras.models.load_model(model_path)
            load_data = np.array(load_data)
        else :
            load_data = pd.read_csv('900_Res_240703_1h_total.csv') #red_0213.csv
            model_path = file_path + '__' + str(j) + "_0710_rctl_model.h5"
            load_model = keras.models.load_model(model_path)
            load_data = np.array(load_data)

    
        call_data = data[['call_mb']]
        sms_data = data[['mms_mb']]
        internet_data = data['internet']
        combined_data = pd.concat([call_data, sms_data, internet_data], axis=1)

        overall_min = combined_data.min().min()  # 모든 열의 최소값들 중 최소값
        overall_max = combined_data.max().max()  # 모든 열의 최대값들 중 최대값

        global_min = combined_data.min()
        global_max = combined_data.max()

        global_min = np.array(global_min)
        global_max = np.array(global_max)

        global_min = global_min.reshape(-1, 3)
        global_max = global_max.reshape(-1, 3)

        print(global_min.shape)
        
        print(load_data.shape)
        for k in range(load_data.shape[0]): #load_data.shape[0]

            #print(k)
            #cell_id = int(load_data[k]) 
            cell_id = int(load_data[k, 0])

            cell_num = j
            cell_model_num = j

            x_data = data[data.gridID == cell_id][['call_mb']]
            z_data =  data[data.gridID == cell_id][['mms_mb']]
            c_data =  data[data.gridID == cell_id][['internet']]
            y_data =  data[data.gridID == cell_id]['startTime']

            last = pd.concat([y_data, x_data, z_data, c_data], axis = 1)
            last.columns = ['time', 'call', 'mms', 'internet']
            last = last.sort_values( by = 'time', ascending = True)

            x_data = np.array(last)
            x_data = x_data[: , 1:]

            if x_data.shape[0] == 1488: # 행의 개수 = 1488
                train_data = x_data[:24*30, :]
                # xx_data = x_data[:24*30, :] # 11월 1일부터 11월 30일까지의 데이터
                test_data = x_data[24*30: 24*(30+8), :] # 12월 1일부터 12월 8일까지의 데이터

            else :
                pass

            # max 값 추출
            call_max = train_data[:, 0].max()
            mms_max = train_data[:, 1].max()
            internet_max = train_data[:, 2].max()
            
            combined_max_data = pd.DataFrame({
                'Call Max': [call_max],
                'MMS Max': [mms_max],
                'Internet Max': [internet_max]
            })

            #print('combined_max_data', combined_max_data)

            # 정규화(min-max)
            min = np.min(x_data)
            max = np.max(x_data)


            train_data = normalize(train_data, overall_min, overall_max)
            test_data = normalize(test_data, overall_min, overall_max)

            # 예측하기 위한 준비 (시퀀스별로 slicing)
            # Train
            steps = 3
            # for i in range(steps, train_data.shape[0]-steps):
            #     x_train = np.append(x_train, train_data[i-steps:i, :]) # 지난 시퀀스 데이터 (i 직전까지)
            #     y_train = np.append(y_train, train_data[i, :])         # 타겟 (예측할 )

            # Test
            for i in range(steps, test_data.shape[0]):
                x_test = np.append(x_test, test_data[i-steps:i, :])
                y_test = np.append(y_test, test_data[i, :])
                #print(x_test.shape)

        # x_train, y_train = np.array(x_train), np.array(y_train)
        x_test, y_test = np.array(x_test), np.array(y_test)
        print('--------------------')
        print(x_test.shape) # 3 sequence의 트래픽 데이터 (call, mms, internet) 
        print('--------------------')
        # print(len(train_data))
        # print(len(test_data))

        # 인덱스 배열 생성
        # idx = np.arange(x_train.shape[0])
        # idx = np.random.shuffle(idx)
        # np.random.shuffle(idx)

        # 데이터 섞기
        # x_train = x_train[idx, :, :]
        # y_train = y_train[idx, :]

        # x_train = x_train[0]
        # y_train = y_train[0]

        # x_train = x_train.reshape(-1, steps, 3)
        # y_train = y_train.reshape(-1, 3)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        # inx_x = np.arange(0, x_train.shape[0], 3)

        # inx_x = random.shuffle(inx_x)

        print(x_test.shape)
        # print(y_train.shape)
        # print(y_train)
        # x_train = x_train.astype(np.float32)
        # y_train = y_train.astype(np.float32)
        # x_test = x_test.astype(np.float32)
        x_test = np.array(x_test, dtype=np.float32)
        # y_test = y_test.astype(np.float32) 

        # keras 모델 불러오기
        #file_path = '3DATA_0208/3DATA_ALL/internet_1hour_all_cells_new_data_step_3'
        #C:\Users\nayeon\Desktop\연구\dataset\traffic_prediction\3DATA_0208\3DATA_ALL\internet_1hour_all_cells_new_data_step_3__0_conv1_tcn_lstm_model.keras

        #predictions = []
        # for j in range(2):
                
        #         # 모델 로드
        #         model_path = file_path + '__' + str(j) + "_conv1_tcn_lstm_model.keras"
        #         load_model = keras.models.load_model(model_path)
        #         # if j == 0  :
        #         #     print('business')
        #         #     #load_model = np.array(load_model)
        #         # else :
        #         #     print('residential')
        #         #     # load_model = np.array(load_model)
                
        #         load_model.summary()

            # 모델 예측
        prediction = np.sum(load_model.predict(x_test), axis=1)
                # print('predictions', prediction)
                # print(len(prediction))
        predictions.append(prediction)

        y_test = np.sum(y_test, axis=1)
        y_test_list.append(y_test)
        
                
    # loss=11%이므로 예측값에 1.1 곱하기
    predictions = np.array([predictions])
    predictions *= 1.1
    
    predictions = denormalize(predictions, overall_min, overall_max)
    predictions = predictions.flatten()

    y_test_list = np.array([y_test_list])
    y_test_list = denormalize(y_test_list, overall_min, overall_max)
    y_test_list = y_test_list.flatten()

    print(y_test_list.shape, predictions.shape)
    

    print('predictions', predictions[0])
    # print('1st predicted', prediction[:3])
    print('1st GT', y_test_list)

    plot_data = pd.DataFrame({
        'ground_truth' : y_test_list, 
        'predictions': predictions
        })
    plot_data.to_csv('results_0710.csv', index=False)
