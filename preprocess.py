#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：swin_tranformer 
@File    ：preprocess.py.py
@IDE     ：PyCharm 
@Author  ：yyd
@Date    ：2024/5/9 17:27 
@Task    : 数据处理
'''
import base64
import cv2
import numpy as np
import json
import requests
from io import BytesIO


class ImageProcessor:
    def __init__(self, json_data):
        # self.json_file = json_file
        self.data = json.loads(json_data)
    
    def requests_get(self, images_url):
        images = []
        for image_url in images_url:
            response = requests.get(image_url)
            response.raise_for_status()
            image_bytes = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            images.append(image)
        return images
    
    def crop_contour(self, image, points):
        """Crop the image based on the contour points."""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)  # 创建一个与图像大小相同的零矩阵
        contour = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [contour], 255)  # 填充多边形区域为255
        cropped_image = cv2.bitwise_and(image, image, mask=mask)  # 应用掩码
        x, y, w, h = cv2.boundingRect(contour)
        return cropped_image[y:y + h, x:x + w]  # 裁剪图像

    def process_images(self):
        """Process all images and contours specified in the loaded JSON data."""
        images_url = self.data['param']['image']
        # images_param = self.data["param"]["image"]
        contours = self.data['param']['shapes']
        contours_crops = []  # List to hold all contours crops across all images
        images_list = self.requests_get(images_url)
        for contour in contours:
            contour_crops = []  # List to hold crops of this contour from each image
            # for image_param in images_param:
            for image in images_list:
                # if image is None:
                #     raise FileNotFoundError(f"Image not found.")
                cropped = self.crop_contour(image, contour['points'])
                contour_crops.append(cropped)
            contours_crops.append([contour['id'], contour_crops])

        return contours_crops


# 主函数
if __name__ == "__main__":
    
    json_file = r'G:\baidu_docker\swin_tranformer\data\output_file.json'
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
    json_data = json.dumps(json_data)
    image_processor = ImageProcessor(json_data)
    contours_crops_array = image_processor.process_images()
    print(contours_crops_array)
