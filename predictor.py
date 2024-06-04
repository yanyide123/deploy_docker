import torch
import onnxruntime as rt
from mmcv import Config
import os
import json
from preprocess import ImageProcessor
from mmaction.apis import inference_recognizer_pth
from mmaction.apis import inference_recognizer_onnx  # Assuming this is the correct import path
from mmaction.apis import init_recognizer, inference_recognizer


class ModelPredictor:
    def __init__(self, config_path, model_path):
        # Choose the device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Set provider type for the executor
        providers = ['CUDAExecutionProvider'] if self.device == "cuda:0" else ['CPUExecutionProvider']
        # Load ONNX model
        # self.ort_session = rt.InferenceSession(model_path, providers=providers)
        # Load configuration from file
        self.cfg = Config.fromfile(config_path)
        self.model = init_recognizer(config_path, model_path, device='cuda:0')

    def predict(self, contours_crops_array):
        results = []
        # for tensor, video_path in unrecognized_list:
        for tensor_point in contours_crops_array:
            tensor = tensor_point[-1]
            point_id = tensor_point[0]
            video_path = "ll.mp4"
            # Use ONNX to perform inference using mmaction's inference_recognizer_onnx
            # outputs_class, outputs_score = inference_recognizer_onnx(self.cfg, video_path, tensor, self.ort_session)
            outputs_class, outputs_score = inference_recognizer_pth(self.cfg, video_path, tensor, self.model)
            results.append({
                "id": point_id,
                "label": outputs_class,
                "score": outputs_score
            })
        return results


# 使用例子
if __name__ == "__main__":


    def class_dict_53(key):
        class_dict_list = \
            {
                0: '石英', 1: '燧石', 2: '长石', 3: '花岗岩', 4: '中基性喷出岩', 5: '酸性喷出岩', 6: '高变岩', 7: '石英岩', 8: '片岩',
                9: '千枚岩', 10: '板岩', 11: '变质砂岩', 12: '粉砂岩', 13: '泥岩', 14: '白云岩', 15: '灰岩', 16: '白云母', 17: '黑云母',
                18: '其他云母', 19: '绿泥石', 20: '火山碎屑岩', 21: '中基性侵入岩', 22: '透明重矿物', 23: '不透明重矿物', 24: '泥化碎屑',
                25: '钙化碎屑', 26: '云化碎屑', 27: '硅化陆屑', 28: '菱铁矿化陆屑', 29: '其他陆屑', 30: '高岭石', 31: '水云母', 32: '绿泥石',
                33: '网状粘土', 34: '蒙脱石', 35: '凝灰质', 36: '方解石', 37: '铁方解石', 38: '白云石', 39: '铁白云石', 40: '菱铁矿',
                41: '硬石膏', 42: '石膏', 43: '重晶石', 44: '浊沸石', 45: '硅质', 46: '长石质', 47: '黄铁矿', 48: '泥铁质', 49: '碳酸盐杂基',
                50: '其他杂基', 51: '其他胶结物', 52: '非陆源碎屑',
            }
        value = class_dict_list[int(key)]
        return value


    config_file = r'G:\baidu_docker\swin_tranformer\weights\cq\53.py'
    # download the checkpoint from model zoo and put it in `checkpoints/`
    model_file = r'G:\baidu_docker\swin_tranformer\weights\cq\best_top1_acc_epoch_25.pth'
    # config_file = r"G:\baidu_docker\swin_tranformer\weights\cq\cq_53_config.py"
    # model_file = r"G:\baidu_docker\swin_tranformer\weights\cq\yyd_cq_best_top1_acc_epoch_26.onnx"
    predictor = ModelPredictor(config_file, model_file)
    json_file = r'G:\baidu_docker\swin_tranformer\data\result.json'
    def load_json(json_file):
        """Load JSON data from a file."""
        if json_file.startswith('"') and json_file.endswith('"'):
            clean_path = json_file[1:-1]
        else:
            clean_path = json_file
        with open(clean_path, 'r') as file:
            data = json.load(file)
        return data
    json_data = load_json(json_file)
    image_processor = ImageProcessor(json_data)
    contours_crops_array = image_processor.process_images()
    result_list = predictor.predict(contours_crops_array)
    print(len(result_list))
    for results in result_list:
        print("岩石的id {},岩性是{},分类的置信度{}".format(results["id"], class_dict_53(results["label"]), results["score"]))
