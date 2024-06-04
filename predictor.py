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
    0: '类别1', 1: '类别2', 2: '类别3', 3: '类别4', 4: '类别5', 5: '类别6', 6: '类别7', 7: '类别8', 8: '类别9',
    9: '类别10', 10: '类别11',
}
        value = class_dict_list[int(key)]
        return value


    config_file = './weights/cq/53.py'
    # download the checkpoint from model zoo and put it in `checkpoints/`
    model_file = './weights/cq/best_top1.pth'
    predictor = ModelPredictor(config_file, model_file)
    json_file = './data/result.json'
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
        print("的id {},是{},分类的置信度{}".format(results["id"], class_dict_53(results["label"]), results["score"]))
