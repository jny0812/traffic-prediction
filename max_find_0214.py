from msvcrt import kbhit
from tabnanny import verbose
import pandas as pd
import sys
import numpy as np
import glob
import keras
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
import tensorflow as tf
import tensorflow_addons as tfa
import time
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import inspect
from typing import List
from keras import backend as K, Model, Input, optimizers
import tensorflow as tf
from sklearn.metrics import mean_squared_error
#from . import TCN, tcn_full_summary
from tcn import TCN, tcn_full_summary
from sklearn.metrics import mean_squared_error
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.models import Sequential
from keras.layers import Dense, GRU, Dropout, Conv1D, BatchNormalization, Activation, ReLU, Add, Flatten, concatenate ,multiply
from keras.models import Model, load_model
import tensorflow_addons as tfa
import csv
import glob
import os
import math
import random
import pydot
from keras.utils import plot_model
from attention import Attention
from keras_self_attention import SeqSelfAttention
from silence_tensorflow import silence_tensorflow
from collections import Counter
silence_tensorflow()
with tf.device('/GPU:0'):
        b_0_time = []
        b_1_time = []
        b_2_time = []
        b_3_time = []
        b_4_time = []
        b_5_time = []
        b_6_time = []
        b_7_time = []
        b_8_time = []
        b_9_time = []
        b_10_time = []
        b_11_time = []
        b_12_time = []
        b_13_time = []
        b_14_time = []
        b_15_time = []
        b_16_time = []
        b_17_time = []
        b_18_time = []
        b_19_time = []
        b_20_time = []
        b_21_time = []
        b_22_time = []
        b_23_time = []
        bu_list = []
        red_list = []
        list_05 = []
        list_611 = []
        list_1217 = []
        list_1823 = []
        data = pd.read_csv('milan_raw_data.csv')
        cell_list = pd.read_csv('900_cells.csv')
        cell_ilist = np.array(cell_list)
        num = 0
        ans = np.zeros((1, 24))
        for j in range(cell_ilist.shape[0]):
        #for j in range(1, 10001):
            test_list = []
            cell_id = cell_ilist[j]
            cell_id = int(cell_id)
            steps = 6
            x_data = data[data.CellID == cell_id]['internet']
            y_data =  data[data.CellID == cell_id]['hour'] + data[data.CellID == cell_id]['weekday'] * 24

            last = pd.concat([x_data, y_data], axis = 1)
            last.columns = ['internet', 'time']
            last = last.sort_values( by = 'time', ascending = True)
            x_data = np.array(last)

            x_data = x_data[:, 0]
            x_data = x_data.reshape(-1, 1)

            if x_data.shape[0] < 720 :
                #print(x_data.shape)
                print(cell_id)
            else : 
                pr_data = x_data[3*24+1: 4*24+1]
                test_list.append(int(np.argmax(pr_data)))
                # 11월 1일이 금요일임
                for j in range(4, 31):
                    range_v = 0
                    if j % 7 != 1 and j % 7 != 2 :
                        #print(j)
                        num += 1
                        #print(pr_data)
                        pr_data = x_data[j*24+1 : (j+1)*24+1]
                        test_list.append(int(np.argmax(pr_data)))
                #print(pr_data.shape)


                counter = Counter(test_list)
                most_common_element, freq = counter.most_common(1)[0]
                # Use the most common element from test_list for 'a'
                a = most_common_element
                ans[0, a] += 1

            if a >= 15 or a <= 1:
                b_0_time.append(cell_id)
            else:
                b_1_time.append(cell_id)
        print(ans)
        b_time = np.array(b_0_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('900_Bus_240703_1h_total.csv', index=False)

        b_time = np.array(b_1_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('900_Res_240703_1h_total.csv', index=False)

        '''
                if a == 0 :

                    b_0_time.append(cell_id)
                elif a == 1 :
                    b_1_time.append(cell_id)
                elif a == 2 :
                    b_2_time.append(cell_id)
                elif a == 3 :
                    b_3_time.append(cell_id)     
                elif a == 4 :
                    b_4_time.append(cell_id)
                elif a == 5 :
                    b_5_time.append(cell_id)
                elif a == 6 :
                    b_6_time.append(cell_id)     
                elif a == 7 :
                    b_7_time.append(cell_id)
                elif a == 8 :
                    b_8_time.append(cell_id)
                elif a == 9 :
                    b_9_time.append(cell_id)
                elif a == 10 :
                    b_10_time.append(cell_id)
                elif a == 11 :
                    b_11_time.append(cell_id)
                elif a == 12 :
                    b_12_time.append(cell_id)     
                elif a == 13 :
                    b_13_time.append(cell_id)
                elif a == 14 :
                    b_14_time.append(cell_id)
                elif a == 15 :
                    b_15_time.append(cell_id)     
                elif a == 16 :
                    b_16_time.append(cell_id)
                elif a == 17 :
                    b_17_time.append(cell_id)
                elif a == 18 :
                    b_18_time.append(cell_id)
                elif a == 19 :
                    b_19_time.append(cell_id)
                elif a == 20 :
                    b_20_time.append(cell_id)
                elif a == 21 :
                    b_21_time.append(cell_id)     
                elif a == 22 :
                    b_22_time.append(cell_id)
                elif a == 23 :
                    b_23_time.append(cell_id)
        print(ans)

        ward = '0127'
        b_time = np.array(b_0_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_0.csv', index=False)

        b_time = np.array(b_1_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_1.csv', index=False)

        b_time = np.array(b_2_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_2.csv', index=False)

        b_time = np.array(b_3_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_3.csv', index=False)

        b_time = np.array(b_4_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_4.csv', index=False)

        b_time = np.array(b_5_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_5.csv', index=False)

        b_time = np.array(b_6_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_6.csv', index=False)

        b_time = np.array(b_7_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_7.csv', index=False)

        b_time = np.array(b_8_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_8.csv', index=False)

        b_time = np.array(b_9_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_9.csv', index=False)

        b_time = np.array(b_10_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_10.csv', index=False)

        b_time = np.array(b_11_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_11.csv', index=False)

        b_time = np.array(b_12_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_12.csv', index=False)

        b_time = np.array(b_13_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_13.csv', index=False)

        b_time = np.array(b_14_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_14.csv', index=False)
        
        b_time = np.array(b_15_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_15.csv', index=False)

        b_time = np.array(b_16_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_16.csv', index=False)

        b_time = np.array(b_17_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_17.csv', index=False)

        b_time = np.array(b_18_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_18.csv', index=False)

        b_time = np.array(b_19_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_19.csv', index=False)

        b_time = np.array(b_20_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_20.csv', index=False)

        b_time = np.array(b_21_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_21.csv', index=False)

        b_time = np.array(b_22_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_22.csv', index=False)

        b_time = np.array(b_23_time)
        b_time  = pd.DataFrame(b_time)
        b_time.to_csv('1+900_23.csv', index=False)
        '''