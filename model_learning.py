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

silence_tensorflow()
with tf.device('/GPU:1'):

    # 정규화 함수 정의
    def normalize(df, global_min, global_max):
        return (df - global_min) / (global_max - global_min)

    #gelu adamw
    random.seed(53)
    np.random.seed(53)
    cell_id = 1
    steps = 8
    batch_size = 1024
    epochs = 5
    
    # trainm test 데이터셋 & 엣지서버 기준값 불러오기
    data_path = 'C:\\Users\\nayeon\\Desktop\\research\\edge_server\\datasets'
    data = pd.read_csv(data_path + '\\hourlyGridActivity.csv')

    # 전처리
    data = data_load.redifine_types_to_mb(data)

    for i in range(2, 3):
        steps = 3
        if i == 0:
            word = 'sms'
        if i == 1:
            word = 'call'
        else:
            word = 'internet'
        file_path = '3DATA_0208/3DATA_ALL/' + word + '_0710_1hour_all_cells_new_data_step_3_'
        ans = np.zeros((100, 100))
        cell_model_num = i
        time_steps = steps
        input_dim = 1
        f_list = [16, 32, 64, 64, 32, 16]

        # 학습 함수 정의
        def tcn_lstm_1(x_train, y_train, x_test, y_test, cell_num, time_steps, batch_size=512, epochs=1000):
            # conv tcn
            inputs = Input(shape=(time_steps, 3), dtype=tf.float32, name='input_1')
            # block 1 (16)
            c1 = 16
            fx = Conv1D(c1, 3, padding='causal', dilation_rate=1)(inputs)
            fx = BatchNormalization()(fx)
            fx = ReLU()(fx)
            fx = Dropout(0.05)(fx)
            fx = Conv1D(c1, 3, padding='causal', dilation_rate=1)(fx)
            fx = BatchNormalization()(fx)
            fx = ReLU()(fx)
            d2 = Dropout(0.05)(fx)
            inx = Conv1D(c1, 1, padding='causal', dilation_rate=1)(inputs)
            # block의 output과 입력에 1x1 conv한 것을 더함
            outputs_c1 = Add()([inx, d2])
            outputs_c1 = LSTM(return_sequences=True, units=c1)(outputs_c1)
            # 초기 입력에 필터 크기만큼의 1x1 conv
            outputs_c1_f = Conv1D(c1, 1, padding='same', dilation_rate=1)(inputs)
            # block의 최종 output과 input과의 더함
            outputs_c1 = Add()([outputs_c1_f, outputs_c1])
            output_16 = Conv1D(16, 1, padding='same', dilation_rate=1)(outputs_c1)

            for j in range(1, len(f_list)):
                # block 2
                c2 = f_list[j]
                # 첫번째 블록의 최종 output(이후 이 for문의 output)을 입력으로 함
                fx = Conv1D(c2, 3, padding='causal', dilation_rate=2 * j)(outputs_c1)
                fx = BatchNormalization()(fx)
                fx = ReLU()(fx)
                fx = Dropout(0.05)(fx)
                fx = Conv1D(c2, 3, padding='causal', dilation_rate=2 * j)(fx)
                fx = BatchNormalization()(fx)
                fx = ReLU()(fx)
                d2 = Dropout(0.05)(fx)
                # 입력으로 들어온 값에 1x1 conv를 통해 채널 맞춤
                inx = Conv1D(c2, 1, padding='causal', dilation_rate=1)(outputs_c1)
                # 출력은 1x1 후 값과 blcok의 output을 합침
                outputs_c2 = Add()([inx, d2])
                outputs_c2 = LSTM(return_sequences=True, units=c2)(outputs_c2)
                # 이후 최초 입력 값에 1x1 conv 진행
                outputs_c2_f = Conv1D(c2, 1, padding='same', dilation_rate=1)(outputs_c1)
                # block에 최종 output과 블록 초기 값을 더함
                outputs_c2 = Add()([outputs_c2_f, outputs_c2])
                # print(outputs_c2.shape)
                # 추가적인 더할 값 저장
                if j == 1:
                    output_32 = Conv1D(32, 1, padding='same', dilation_rate=1)(outputs_c2)
                    outputs_c1 = outputs_c2
                if j == 2:
                    output_64 = Conv1D(64, 1, padding='same', dilation_rate=1)(outputs_c2)
                    outputs_c1 = outputs_c2
                # 다음 block 입력으로 현 blcok의 output이 들어감
                if j == 3:
                    outputs_c1 = Add()([output_64, outputs_c2])
                if j == 4:
                    outputs_c1 = Add()([output_32, outputs_c2])
                if j == 5:
                    outputs_c1 = Add()([output_16, outputs_c2])

            # 가장큰 level 1의 값을 더함
            ori_in = Conv1D(16, 1, padding='same', dilation_rate=1)(inputs)
            ori_in = Add()([outputs_c1, ori_in])
            outputs_c3 = Flatten()(ori_in)
            outputs = Dense(3)(outputs_c3)

            # 모델 정의
            model = Model(inputs=inputs, outputs=outputs, name='multi_red')
            # model.summary()

            # 모델 컴파일 (손실 함수 - MAPE)
            model.compile(optimizer=tf.optimizers.Adam(), loss='MAE')
            callbacks_tcn = [
                keras.callbacks.ModelCheckpoint(filepath=file_path + '_' + str(cell_num) + '_0710_rctl_model.h5',  # h5->keras로 수정
                                                monitor='val_loss',
                                                save_best_only=True,
                                                )]
            start_new = time.time()  # 시작 시간 저장

            # 모델 학습
            history_tcn = model.fit(
                x_train,
                y_train,
                epochs=epochs,
                validation_data=(x_test, y_test),
                callbacks=callbacks_tcn,
                batch_size=batch_size,
                verbose=1
            )

            # 학습 손실 값 출력
            print("Training Loss per epoch:", history_tcn.history['loss'])
            # 검증 손실 값 출력 (검증 데이터가 있는 경우)
            if 'val_loss' in history_tcn.history:
                print("Validation Loss per epoch:", history_tcn.history['val_loss'])

            # 모델 예측 (MAPE loss에 의해 1.1을 곱해줌)
            # predictions = model.predict(x_test) * 1.1

            new_time = time.time() - start_new  # 현재시각 - 시작시간 = 실행 시간

            return history_tcn, new_time, model

        stack_train = []
        stack_test = []

        traffics = []

        for j in range(2):
            if j == 0:
                load_data = pd.read_csv('900_Bus_240703_1h_total.csv')  # bu_0213.csv
                print('a')
                load_data = np.array(load_data)
            else:
                load_data = pd.read_csv('900_Res_240703_1h_total.csv')  # red_0213.csv
                print('b')
                load_data = np.array(load_data)
            print(load_data.shape)

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

            x_train = []
            y_train = []
            x_test = []
            y_test = []

            # max 데이터 값 저장 변수
            combined_max_data = pd.DataFrame(columns=['call_max', 'mms_max', 'internet_max'])

            for k in range(load_data.shape[0]):  # load_data.shape[0]
                # cell_id = int(load_data[k])
                cell_id = int(load_data[k, 0])

                cell_num = j
                cell_model_num = j

                x_data = data[data.gridID == cell_id][['call_mb']]
                z_data = data[data.gridID == cell_id][['mms_mb']]
                c_data = data[data.gridID == cell_id][['internet']]
                y_data = data[data.gridID == cell_id]['startTime']

                last = pd.concat([y_data, x_data, z_data, c_data], axis=1)
                last.columns = ['time', 'call', 'mms', 'internet']
                last = last.sort_values(by='time', ascending=True)

                x_data = np.array(last)
                x_data = x_data[:, 1:]

                if x_data.shape[0] == 1488:  # 행의 개수 = 1488
                    train_data = x_data[:24 * 30, :]
                    # xx_data = x_data[:24*30, :] # 11월 1일부터 11월 30일까지의 데이터
                    test_data = x_data[24 * 30: 24 * (30 + 8), :]  # 12월 1일부터 12월 8일까지의 데이터
                else:
                    pass

                # max 값 추출
                call_max = train_data[:, 0].max()
                mms_max = train_data[:, 1].max()
                internet_max = train_data[:, 2].max()

                max_value_df = pd.DataFrame({
                    'call_max': [call_max],
                    'mms_max': [mms_max],
                    'internet_max': [internet_max]
                })

                combined_max_data = pd.concat([combined_max_data, max_value_df], ignore_index=True)

                # 정규화(min-max)
                min_val = np.min(x_data)  # 변수 이름 변경
                max_val = np.max(x_data)  # 변수 이름 변경

                train_data = normalize(train_data, overall_min, overall_max)
                test_data = normalize(test_data, overall_min, overall_max)

                # 예측하기 위한 준비 (시퀀스별로 slicing)
                # Train
                for idx in range(steps, train_data.shape[0] - steps):
                    x_train.append(train_data[idx - steps:idx, :])  # 지난 시퀀스 데이터 (idx 직전까지)
                    y_train.append(train_data[idx, :])  # 타겟 (예측할 )

                # Test
                for idx in range(steps, test_data.shape[0]):
                    x_test.append(test_data[idx - steps:idx, :])
                    y_test.append(test_data[idx, :])

            # (train) 각 열의 max를 합하여 총 traffic max값 구하기
            call_max_max = combined_max_data['call_max'].max()
            mms_max_max = combined_max_data['mms_max'].max()
            internet_max_max = combined_max_data['internet_max'].max()
            traffics.append(call_max_max + mms_max_max + internet_max_max)

            x_train, y_train = np.array(x_train), np.array(y_train)
            x_test, y_test = np.array(x_test), np.array(y_test)
            print('--------------------')
            print(x_train.shape)  # 3 sequence의 트래픽 데이터 (call, mms, internet)
            print('--------------------')
            print(len(train_data))
            print(len(test_data))

            # 인덱스 배열 생성
            idx = np.arange(x_train.shape[0])
            np.random.shuffle(idx)

            # 데이터 섞기
            x_train = x_train[idx, :, :]
            y_train = y_train[idx, :]

            x_train = x_train.reshape(-1, steps, 3)
            y_train = y_train.reshape(-1, 3)

            print(x_train.shape)
            print(y_train.shape)
            print(y_train)
            x_train = x_train.astype(np.float32)
            y_train = y_train.astype(np.float32)
            x_test = x_test.astype(np.float32)
            y_test = y_test.astype(np.float32)

            multi_tcn_history, multi_tcn_time, multi_tcn = tcn_lstm_1(x_train, y_train, x_test, y_test, cell_model_num, time_steps=steps, epochs=1000)

    # 리스트에서 최대값 찾기
    traffic_max_value = max(traffics)

    # 최대값을 정수로 변환
    traffic_max_int = int(traffic_max_value)

    print(f"The maximum value in traffics is: {traffic_max_int}")
