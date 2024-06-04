#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：swin_tranformer 
@File    ：callback.py
@IDE     ：PyCharm 
@Author  ：yyd
@Date    ：2024/5/23 16:33 
@Task    : 回调函数，由于模型处理时间长，所以需要一个回调函数，下面是一个回调函数的demo
'''
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/callback', methods=['POST'])
def callback_route():
    try:
        # 获取回调数据
        callback_data = request.get_json()
        # 在这里可以处理接收到的回调数据，比如保存到数据库，日志记录等
        print("Received callback data:", callback_data)

        # 返回成功响应
        return jsonify({'status': 'success', 'message': 'Callback data received successfully', 'code': 200})

    except Exception as e:
        # 处理可能出现的错误
        return jsonify({'error': str(e), 'status': False}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
