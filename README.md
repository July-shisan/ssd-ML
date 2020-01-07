# ssd-ML
目标检测任务

训练模型为SSD

基础网络结构为VGG16

train.py为训练部分代码

eval_5epoch_for.py为评估部分代码

calculate_map_test.py为 MAP计算部分代码

test.py为测试部分代码

运行环境为anaconda 4.8 + python3.7 + pytorch 1.2(cuda) + opencv2

运行train.py用VGG网络训练90000次，将训练之后的模型保存为weights/ssd_90000.pth

运行eval_5epoch_for.py对评估数据集进行测试，评估结果保存在eval/results中

运行test.py将测试的预测结果保存在test/中

数据保存在data/VOCdevkit/文件夹中，Annotation存放目标标注信息，ImageSets/Main保存训练集、评估集和测试集图片名称，JPEGImages存放训练图片

测试结果：

~~~~~~~~
Mean AP = 0.9519
Results:
0.942
0.962
0.952

~~~~~~~~
