"""
The script for ape read and save config file
Author: yanbingzheng@baidu.com
Date: 2019/07/19
"""
import argparse
import os
from urllib.parse import urljoin
import pandas as pd
from flask import Flask, request, json, jsonify
from model import PyModel
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--port')
args = parser.parse_args()

pymodel = PyModel()
pymodel.load()

app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():
    """
    预测接口
    :return:
    """
    #请求处理
    df = pd.DataFrame()
    hasConverter = hasattr(pymodel, 'converter')
    if hasConverter:
        #调用 converter 对请求数据做行列转换
        hasDecode = hasattr(pymodel.converter, 'decode')
        if hasDecode:
            if isinstance(request.get_json(), dict):
                df = pymodel.converter.decode(json.dumps(request.get_json()))
            elif isinstance(request.get_data(), bytes):
                df = pymodel.converter.decode(request.get_data(as_text=True))
            else:
                df = pymodel.converter.decode(request.get_json())
        else:
            raise Exception("NameError: name 'decode' is not defined in Pymodel")

    else:
        #兼容旧版本 pymodel，逻辑不变
        json_data = json.loads(request.get_data(as_text=True))
        data = json_data['data']
        data = json.dumps(data)
        df = pd.read_json(data, dtype=False)
    print(df)
    try:
        #预测函数入口
        prediction = pymodel.swin_transform(df)
        # 将 JSON 字符串解析为字典
        prediction_data = json.loads(prediction)
        # 添加新的字段
        prediction_data['status'] = str(True)
        
        # 将字典重新序列化为 JSON 字符串
        json_string_with_status = json.dumps(prediction_data, ensure_ascii=False)
    except Exception as e:
        error_data = json.loads(df)
        # 添加新的字段
        error_data['status'] = str(False)
        error_data['message'] = str(e)
        # 将字典重新序列化为 JSON 字符串
        json_string_with_status = json.dumps(error_data, ensure_ascii=False)

    # 获取回调地址
    # 这里是在docker环境变量里面获取回调地址的
    # example_variable = os.environ.get('HOME')
    # endpoint = 'falcon/identify/identify_model_callback'
    # 下面是测试回调地址
    endpoint = 'callback'
    example_variable = "http://localhost:5001"
    if example_variable:
        callback_url = urljoin(example_variable, endpoint)
    else:
        callback_url = None
        print('EXAMPLE_VARIABLE is not set.')

    # 发送 POST 请求到回调地址
    callback_response = requests.post(callback_url, json=json_string_with_status)

    # 检查回调请求的响应
    callback_response = callback_response.json()
    print(callback_response)
    if callback_response.get('code') == 200:
        return jsonify({'status': 'success', 'message': 'Result sent to callback URL', 'callback_status': True})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to send result to callback URL',
                        'callback_response': callback_response, 'callback_status': False}), 500

    #结果处理，下面是不需要回调，模型秒处理，直接调用直接返回
    # if hasConverter:
    #     #调用 converter 对返回数据做行列转换
    #     hasEncode = hasattr(pymodel.converter, 'encode')
    #     if hasEncode:
    #         return pymodel.converter.encode(json_string_with_status)
    #     else:
    #         raise Exception("NameError: name 'encode' is not defined in Pymodel")
    # else:
    #     #兼容旧版本 pymodel，逻辑不变
    #     res = prediction.to_dict('index')
    #     json_res = json.dumps(res)
    #     res.clear()
    #     print(json_res)
    #     return json_res
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=args.port, threaded=False, debug=False)
