import json
from predictor import ModelPredictor
from preprocess import ImageProcessor
from config.model_config import MODEL_PATHS, MODEL_CLASSES, MODEL_CONFIG_PATH


class CustomConverter(object):
    def encode(self,response):
        return response

    def decode(self,request):
        return request


class ModelService:
    def __init__(self, json_data):
        self.json_data = json_data

    # 模型出来结果进行汇总，把模型结果写入读取的json中
    def update_json_with_predictions(self, predictions):
        # 将预测的结果更新到JSON数据中
        contour_dict = {contour['id']: contour for contour in self.json_data['param']['shapes']}

        for prediction in predictions:
            contour_id = prediction['id']
            if contour_id in contour_dict:
                contour_dict[contour_id]['label'] = MODEL_CLASSES[int(prediction['label'])]
                contour_dict[contour_id]['score'] = str(prediction['score'])
        # 重新生成contours列表
        self.json_data['param']['shapes'] = list(contour_dict.values())
        return self.json_data


class PyModel(object):
    """
    Sklearn 自定义算法代码
    """

    def __init__(self):
        """
        类的构造函数
        """
        self.converter = CustomConverter()
        # 获取模型的路径和配置文件
        self.model = ModelPredictor(MODEL_CONFIG_PATH["model_config_CQ"], MODEL_PATHS["model_CQ_pth"])
        self.class_dict = MODEL_PATHS

    def load(self):
        """
        load the real model
        :return:
        """

    def swin_transform(self, dataset):  # type:(pd.DataFrame)->pd.DataFrame
        """

        :param dataset:
        :return:
        """
        image_processor = ImageProcessor(dataset)
        contours_crops_array = image_processor.process_images()
        result_list = self.model.predict(contours_crops_array)
        data = json.loads(dataset)
        model_service = ModelService(data)
        json_string = model_service.update_json_with_predictions(result_list)
        json_string = json.dumps(json_string)
        return json_string

