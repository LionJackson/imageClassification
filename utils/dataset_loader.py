import keras

"""
加载数据集工具类
"""


class DatasetLoader:
    def __init__(self, path_url, image_size=(224, 224), batch_size=32, class_mode='categorical'):
        self.path_url = path_url
        self.image_size = image_size
        self.batch_size = batch_size
        self.class_mode = class_mode

    # 不使用图像增强
    def load_data(self):
        # 加载训练数据集
        train_data = keras.preprocessing.image_dataset_from_directory(
            self.path_url + '/train',  # 训练数据集的目录路径
            image_size=self.image_size,  # 调整图像大小
            batch_size=self.batch_size,  # 每批次的样本数量
            label_mode=self.class_mode,  # 类别模式：返回one-hot编码的标签
        )

        # 加载验证数据集
        val_data = keras.preprocessing.image_dataset_from_directory(
            self.path_url + '/validation',  # 验证数据集的目录路径
            image_size=self.image_size,  # 调整图像大小
            batch_size=self.batch_size,  # 每批次的样本数量
            label_mode=self.class_mode  # 类别模式：返回one-hot编码的标签
        )
        # 加载测试数据集
        test_data = keras.preprocessing.image_dataset_from_directory(
            self.path_url + '/test',  # 验证数据集的目录路径
            image_size=self.image_size,  # 调整图像大小
            batch_size=self.batch_size,  # 每批次的样本数量
            label_mode=self.class_mode  # 类别模式：返回one-hot编码的标签
        )
        class_names = train_data.class_names
        return train_data, val_data, test_data, class_names
