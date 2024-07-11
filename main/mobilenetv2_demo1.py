import keras
from keras import layers

from utils.dataset_loader import DatasetLoader
from utils.model_util import ModelUtil
from utils.utils import Utils

"""
使用MobileNetV2，实现图像多分类
"""

# 模型训练地址
PATH_URL = '../data/fruits'
# 训练曲线图
RESULT_URL = '../results/fruits'
# 模型保存地址
SAVED_MODEL_DIR = '../saved_model/fruits'

#  图片大小
IMG_SIZE = (224, 224)
# 定义图像的输入形状
IMG_SHAPE = IMG_SIZE + (3,)
# 数据加载批次,训练轮数
BATCH_SIZE, EPOCH = 32, 16


# 训练模型
def train():
    # 实例化数据集加载工具类
    dataset_loader = DatasetLoader(PATH_URL, IMG_SIZE, BATCH_SIZE)
    train_ds, val_ds, test_ds, class_total = dataset_loader.load_data()

    # 构建 MobileNet 模型
    base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False)
    # 将模型的主干参数进行冻结
    base_model.trainable = False
    model = keras.Sequential([
        layers.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SHAPE),
        # 设置主干模型
        base_model,
        # 对主干模型的输出进行全局平均池化
        layers.GlobalAveragePooling2D(),
        # 通过全连接层映射到最后的分类数目上
        layers.Dense(len(class_total), activation='softmax')
    ])
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 模型结构
    model.summary()
    # 指明训练的轮数epoch，开始训练
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCH)
    # 测试
    loss, accuracy = model.evaluate(test_ds)
    # 输出结果
    print('Mobilenet test accuracy :', accuracy, ',loss :', loss)
    # 保存模型 savedModel格式
    model.export(filepath=SAVED_MODEL_DIR)


# 函数式调用方式
def train1():
    # 实例化数据集加载工具类
    dataset_loader = DatasetLoader(PATH_URL, IMG_SIZE, BATCH_SIZE)
    train_ds, val_ds, test_ds, class_total = dataset_loader.load_data()

    inputs = keras.Input(shape=IMG_SHAPE)
    # 加载预训练的 MobileNetV2 模型，不包括顶层分类器，并在 Rescaling 层之后连接
    base_model = keras.applications.MobileNetV3Large(weights='imagenet', include_top=False, input_tensor=inputs)

    # 冻结 MobileNetV2 的所有层，以防止在初始阶段进行权重更新
    for layer in base_model.layers:
        layer.trainable = False
    # 在 MobileNetV2 之后添加自定义的顶层分类器
    x = layers.GlobalAveragePooling2D()(base_model.output)

    predictions = layers.Dense(len(class_total), activation='softmax')(x)
    # 构建最终模型
    model = keras.Model(inputs=base_model.input, outputs=predictions)
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 查看模型结构
    model.summary()
    # 训练模型
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCH)
    # 测试
    loss, accuracy = model.evaluate(test_ds)
    # 输出结果
    print('Mobilenet test accuracy :', accuracy, ',loss :', loss)

    # 保存模型 savedModel格式
    model.export(filepath=SAVED_MODEL_DIR)
    # 保存曲线图
    Utils.trainResult(history, RESULT_URL)


if __name__ == '__main__':
    # train()
    model_util = ModelUtil(SAVED_MODEL_DIR, PATH_URL)
    model_util.batch_evaluation()

