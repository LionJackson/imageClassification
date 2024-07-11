import keras
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

from utils.dataset_loader import DatasetLoader

"""
模型工具类
"""


class ModelUtil:
    def __init__(self, saved_model_dir, path_url):
        self.save_model_dir = saved_model_dir  # savedModel 模型保存地址
        self.path_url = path_url  # 模型训练数据地址

    # 批量识别 进行可视化显示
    def batch_evaluation(self, class_mode='categorical', image_size=(224, 224), num_images=25):
        dataset_loader = DatasetLoader(self.path_url, image_size=image_size, class_mode=class_mode)
        train_ds, val_ds, test_ds, class_names = dataset_loader.load_data()
        # 加载savedModel模型
        tfs_layer = keras.layers.TFSMLayer(self.save_model_dir)
        # 创建一个新的 Keras 模型，包含 TFSMLayer
        model = keras.Sequential([
            keras.Input(shape=image_size + (3,)),  # 根据你的模型的输入形状
            tfs_layer
        ])

        plt.figure(figsize=(10, 10))
        for images, labels in test_ds.take(1):
            # 使用模型进行预测
            outputs = model.predict(images)
            for i in range(num_images):
                plt.subplot(5, 5, i + 1)
                image = np.array(images[i]).astype("uint8")
                plt.imshow(image)
                index = int(np.argmax(outputs[i]))
                prediction = outputs[i][index]
                percentage_str = "{:.2f}%".format(prediction * 100)
                plt.title(f"{class_names[index]}: {percentage_str}")
                plt.axis("off")
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.show()
