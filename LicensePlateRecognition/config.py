'''
参数配置
'''
train_parameters = {
    "input_size": [1, 20, 20],                           #输入图片的shape
    "class_dim": -1,                                     #分类数
    "src_path":"data/data23617/characterData.zip",       #原始数据集路径
    "target_path":"/home/aistudio/data/dataset",        #要解压的路径
    "train_list_path": "./train_data.txt",              #train_data.txt路径
    "eval_list_path": "./val_data.txt",                  #eval_data.txt路径
    "label_dict":{},                                    #标签字典
    "readme_path": "/home/aistudio/data/readme.json",   #readme.json路径
    "num_epochs": 1,                                    #训练轮数
    "train_batch_size": 32,                             #批次的大小
    "learning_strategy": {                              #优化函数相关的配置
        "lr": 0.001                                     #超参数学习率
    }
}