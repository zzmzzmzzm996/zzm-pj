import torch
import pandas as pd
import numpy as np
import threading
import sys
import requests
import logging
import time
import json
import traceback
import matplotlib.pyplot as plt
import os
import joblib
from queue import Queue
# import psutil


# 长时间预测
def predict_test(model, first_set):
    result_list = list(first_set)
    round = future_window
    x = torch.tensor(first_set)
    for i in range(round):
        x = x.to(torch.float32).to(device)
        _, _, res = model(x.unsqueeze(0).unsqueeze(0).to(torch.float32))
        res = torch.flatten(res.detach())
        result_list.extend(np.array(res))
        x = torch.tensor(result_list[-history_window:])
    return np.array(result_list[history_window:])


def exam_offline_predict(path, model):
    global aim_weight
    # raw_data = pd.read_csv(path)
    raw_data = pd.read_csv(path, names=col_names, header=1)
    raw_data.sort_values('Logtime', inplace=True)
    weight_data = np.array(raw_data.loc[:, machine_weight_name])

    model_predict_ave = []
    true_ave = []
    aim_weight_line = []
    adjust_record = []
    sum_of_abs_predict = 0
    sum_of_abs_original = 0

    print(weight_data)
    for i in range(80, 7500):
        aim_weight_line.append(aim_weight)
        # 模拟到达数据
        print('No.', i, 'data has arrived')
        raw_data = np.array(weight_data[:i + future_window])
        aim_weight = float(aim_weight)
        raw_data -= aim_weight
        offset_mean = raw_data.mean()
        offset_std = raw_data.std()

        input_set = (raw_data[-(history_window + future_window):-future_window] - offset_mean) / offset_std

        out_list = predict_test(model, input_set) * offset_std + offset_mean

        out_list = np.concatenate((out_list, input_set[-history_window // 2:]))
        out_ave = out_list.mean()
        model_predict_ave.append(out_ave)
        true_ave.append(raw_data[-(future_window + history_window // 2):].mean())

        print(aim_weight, out_ave + aim_weight, true_ave[-1])
        print('offset_mean:', offset_mean)

        sum_of_abs_predict += abs(out_ave)
        sum_of_abs_original += abs(raw_data[-(future_window + history_window // 2):].mean())

        # TODO 需要一个观察周期

        # 数据实际0.3s来一个，可以设置一个周期进行
        if abs(out_ave) > 1.0:
            # 调整值通常以0.05为最小单位
            suggest_adjust = int(-out_ave / 0.8) * 0.05
            # 先加入缓存
            report_queue.put(suggest_adjust)
        else:
            adjust_record.append(0.0)
        # 调整目标值梯度回归到实际均值
        aim_weight += offset_mean * 0.005
    #
    # plt.figure(figsize=(100, 6.0))
    # ax1 = plt.subplot(2, 1, 1)
    # ax1.plot(range(7500 - 80), model_predict_ave, 'b')
    # ax1.plot(range(7500 - 80), true_ave, 'g')
    # ax2 = ax1.twinx()
    # ax2.plot(range(7500 - 80), adjust_record, 'r')
    # plt.subplot(2, 1, 2)
    # plt.plot(range(7500 - 80), aim_weight_line)
    # plt.show()
    #
    # print('-------------------------\n')
    # print('Offline examination data num:', 7500 - 80)
    # print('Sum of predict offset:', sum_of_abs_predict)
    # print('Sum of original offset:', sum_of_abs_original)
    #
    # true_ave = np.array(true_ave)
    # model_predict_ave = np.array(model_predict_ave)
    # print('Average offset loss:', (true_ave - model_predict_ave).mean())


def send_adjust_value_timer(trigger):
    sum = 0
    que_size = report_queue.qsize()
    # 检查缓存
    logging.info(time.strftime('%y-%m-%d %H:%M:%S') + '\nChecking report cache.\n')

    while not report_queue.empty():
        sum += report_queue.get()
    ave = round(sum / que_size, 2)

    # 向接口发送消息
    headers = {'Content-Type': 'application/json'}
    data = {
        "type": True if ave > 0 else False,
        "res": abs(ave),
        "predict": float(0.00)
    }
    requests.post(url=url, data=json.dumps(data), headers=headers, timeout=10)

    logging.info(time.strftime('%y-%m-%d %H:%M:%S') + '\nSending request success.\n')

    if trigger == 'time':
        threading.Timer(report_time, send_adjust_value_timer, args=[trigger]).start()


def weight_predict(path, model):
    global aim_weight

    raw_data = pd.read_csv(path, names=col_names, header=1)
    weight_data = np.array(raw_data.loc[:, machine_weight_name])

    # 数据数量不足，轮询等待直到数据足够
    if weight_data.shape[0] < history_window:
        logging.info(time.strftime('%y-%m-%d %H:%M:%S') + '\nData num is not enough, waiting for new round...'
                     + '\n-------------- \n\n\n\n')
        timer = threading.Timer(4, weight_predict, args=[path, model])
        timer.start()
        return

    raw_data = np.array(weight_data)

    aim_weight = float(aim_weight)
    # 投入模型的是偏差值
    raw_data -= aim_weight
    # 偏差历史平均值
    offset_mean = raw_data.mean()
    offset_std = raw_data.std()

    input_set = (raw_data[-history_window:] - offset_mean) / offset_std
    out_list = predict_test(model, input_set) * offset_std + coffset_mean

    out_list = np.concatenate((out_list, input_set[-history_window // 2:]))
    out_ave = out_list.mean()
    logging.info(time.strftime('%y-%m-%d %H:%M:%S') + '\nPredicting finished.')

    # 调整目标值梯度回归到实际均值
    # 使用一个调整上界防止数据的突变对整体均值产生影响
    if abs(offset_mean) < 0.7 * 2.0:
        aim_weight += offset_mean * 0.005

    # 预测均值和目前均值差距超过指定值则进行调整
    headers = {'Content-Type': 'application/json'}
    if abs(out_ave) > 1.0:
        # 调整值通常以0.05为最小单位
        suggest_adjust = int(-out_ave / 0.8) * 0.05
        # 尝试加入缓存
        if not report_queue.full():
            report_queue.put(suggest_adjust)
        else:
            send_adjust_value_timer(time='full')

        # data = {
        #     "type": True if suggest_adjust > 0 else False,
        #     "res": abs(suggest_adjust),
        #     "predict": float(out_ave)
        # }
        # requests.post(url=url, data=json.dumps(data), headers=headers, timeout=10)
        # logging.info(time.strftime('%y-%m-%d %H:%M:%S') + '\nSending request success.\n')
    timer = threading.Timer(suggest_time_space, weight_predict, args=[path, model])
    timer.start()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 产生一个建议值的时间间隔（s）
suggest_time_space = 2
# 历史窗口为80
history_window = 80
future_window = 30
# 采样间隔(s)
sampling_time_space = 0.3
# 人工调试时所需的轮训时间（s)
report_time = 5 * 60
# 创建一个调整数据缓存队列
report_queue = Queue(5000)
aim_weight = 0.0
# 接口地址
url = "http://192.168.10.188:8099/business/winson-gage1-adjust-record/adjust"
# 传入数据的列名
col_names = ['Gage1Target', 'Gage2Target', 'Gage3Target', 'Gage1Raw', 'Gage2Raw', 'Gage3Raw', 'Gage1Smooth',
             'Gage2Smooth', 'Gage3Smooth', 'Logtime']
# 需要的参数名（此处为平均克重）
machine_weight_name = 'Gage1Smooth'
predict_water = np.zeros(40)


def predict1(path, model):
    global predict_water
    dataframe = pd.read_csv(path, error_bad_lines=False, engine='python')
    water = np.array(dataframe.iloc[:, 4])
    if water.shape[0] < 60:
        logging.info(time.strftime('%y-%m-%d %H:%M:%S') + '\nData water num is not enough, waiting for new round...'
                     + '\n-------------- \n\n\n\n')
        timer = threading.Timer(4, predict1, args=[path, model])
        timer.start()
        return
    water = water[-60:]
    water_goal = dataframe.iloc[-1, 1]
    water_downsample = []
    for i in range(0, 60, 2):
        sample = min(water[i], water[i + 1])
        water_downsample.append(sample)
    water_downsample = np.array(water_downsample)
    water_run = water_downsample.reshape(1, 30)
    predict_number = model.predict(water_run)
    logging.info(time.strftime('%y-%m-%d %H:%M:%S') + '\nPredicting water finished.')
    predict_water = np.append(predict_water, predict_number[0])
    window = predict_water[-40:]
    high = window[window > water_goal]
    ratio = len(high) / 40
    headers = {'Content-Type': 'application/json'}
    data = {
        "res": ratio
    }
    requests.post(url="http://192.168.10.188:8099/business/winson-gage2-adjust-record/adjust", data=json.dumps(data),
                  headers=headers, timeout=10)
    logging.info(time.strftime('%y-%m-%d %H:%M:%S') + '\nSending request success.')
    timer = threading.Timer(suggest_time_space, predict1, args=[path, model])
    timer.start()


if __name__ == '__main__':
    try:
        file_path = sys.argv[1]
        aim_weight = sys.argv[2]
        model_path = '11.22_lstm_76.0_0.3.pth'

        # 配置日志文件
        logging.basicConfig(level='DEBUG', filename='./logs.txt', filemode='a+')

        # 测试用
        # print('start.py 运行成功')
        # print('file_path', file_path)
        # print('aim_weight', aim_weight)
        # time.sleep(300)

        if not os.path.exists(file_path):
            logging.info(time.strftime('%y-%m-%d %H:%M:%S') +
                         ' file: ' + file_path + ' not exist.Check if it is avaliable.'
                         + '\n-------------- \n\n\n\n')
            sys.exit()

        print("Processing file:", file_path)
        print("Now using", device, "to calculate the results.")
        model = torch.load(model_path, map_location=device)
        model.to(device)
        model_water = joblib.load('model_water.pkl')

        print("Waiting for the model to get the first input set...")
        print('It will be about 30 seconds.')

        # 触发接口发送的情景
        trigger = 'time'
        threading.Timer(suggest_time_space, weight_predict, args=[file_path, model]).start()
        threading.Timer(suggest_time_space, predict1, args=[file_path, model_water]).start()
        threading.Timer(report_time, send_adjust_value_timer, args=[trigger]).start()
    except Exception:
        logging.error(time.strftime('%y-%m-%d %H:%M:%S') + traceback.format_exc() + '-------------- \n\n\n\n')

    # file_path = sys.argv[1]
    # aim_weight = sys.argv[2]
    # model_path = '11.22_lstm_76.0_0.3.pth'
    # model = torch.load(model_path, map_location=device)
    # model.to(device)
    #
    # exam_offline_predict(file_path, model)
