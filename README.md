# swin_transformer

Flask封装分类模型



```
swin_transformer
	|-- client       测试服务，给服务post数据
		|-- client.py     执行文件
	|-- config       配置文件
		|-- model_config.py   模型配置文件
	|-- data         数据文件
	|-- mmaction     模型包
	|-- weights      权重文件夹
	|-- callback.py  回调接口，模型处理结果post到回调接口
	|-- Dockerfile   docker执行创建镜像文件
	|-- model.py     模型代码
	|-- predictor.py 模型预测代码，可以理解为推理代码
	|-- preprocess.py 数据处理代码
	|-- py_model_server.py 封装服务的启动文件
	|-- requirements.txt   环境需要安装的包
	
```

一、Flask封装模型

第一步：py_model_server.py   获取post数据，这个文件不用改动，接收数据类型bytes

第二步：model.py    加载模型，load权重，进行推理文件

第三步：preprocess.py    数据处理代码，模型输入前的处理代码，作为模型输入代码

第四步：model_config.py   模型配置文件，把模型加载的路径、权重路径、模型分类结果对应

第五步：predictor.py 模型预测代码，可以理解为推理代码

二、制作docker镜像

按照规则编写Dockerfile文件，将需要按章的包写到requirements.txt，最好指定版本

编写好执行 docker build -t 镜像名：版本  .     docker build -t my_image .

三、进入镜像测试

1、启动镜像：

docker run --gpus all --network="host" -it -p 8080:8080 --name sss swin_transformer:v1 /bin/bash

2、进入镜像

docker exec -it sss /bin/bash

使用root用户进入镜像

docker exec -u root -it sss /bin/bash

3、启动服务测试

启动服务： python py_model_server.py 封装服务的启动文件

测试数据：python client.py     执行文件传入数据

回调数据接口： python callback.py  回调接口，模型处理结果post到回调接口

4、服务测试成功，提交镜像.tar文件

docker save -o 保存的.tar文件名      镜像名：版本

docker save -o my_image.tar my_image:latest




