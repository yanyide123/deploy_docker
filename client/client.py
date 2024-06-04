import requests
import json
import base64


def load_json(json_file):
    """Load JSON data from a file."""
    if json_file.startswith('"') and json_file.endswith('"'):
        clean_path = json_file[1:-1]
    else:
        clean_path = json_file
    print(clean_path)
    with open(clean_path, 'r') as file:
        print(file)
        data = json.load(file)
    return data

# 将字典转换为 JSON 字符串
# 指定预测方法为lac并发送post请求，content-type类型应指定json方式
url = 'http://localhost:5000/'
json_file = './data/output_file.json'
json_data = load_json(json_file)
print(type(json.dumps(json_data)))
headers = {"Content-Type": "application/json"}
r = requests.post(url=url, headers=headers, data=json.dumps(json_data))

# 打印预测结果
print(json.dumps(r.json(), indent=4, ensure_ascii=False))
# 设置保存文件的路径
output_file_path = './client/prediction_results.json'

# 打开一个文件用于写入
with open(output_file_path, 'w', encoding='utf-8') as file:
    # 使用json.dump将数据写入文件
    json.dump(r.json(), file, indent=4, ensure_ascii=False)