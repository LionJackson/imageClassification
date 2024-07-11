import PIL
import numpy as np
import matplotlib.pyplot as plt
from PIL.Image import Image


class Utils:
    # 生成图片 显示训练曲线图
    @staticmethod
    def trainResult(model, path_url):
        acc = model.history['accuracy']
        val_acc = model.history['val_accuracy']
        loss = model.history['loss']
        val_loss = model.history['val_loss']

        # 按照上下结构将图画输出
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        # 保存图表到本地文件
        plt.savefig(path_url + '.png', dpi=100)
