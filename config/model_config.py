#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：swin_tranformer 
@File    ：model_config.py.py
@IDE     ：PyCharm 
@Author  ：yyd
@Date    ：2024/5/10 9:59 
@Task    : 模型的配置文件
'''

# 模型配置字典
MODEL_PATHS = {
    "model_CQ_pth": "./weights/model/best_acc.pth"
}
MODEL_CONFIG_PATH = {
    "model_config_CQ":  './weights/model/config.py'
}

# 模型分类结果对应
MODEL_CLASSES = {
    0: '类别1', 1: '类别2', 2: '类别3', 3: '类别4', 4: '类别5', 5: '类别6', 6: '类别7', 7: '类别8', 8: '类别9',
    9: '类别10', 10: '类别11',
}

