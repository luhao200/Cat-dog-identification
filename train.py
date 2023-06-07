from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def define_cnn_model():
    # 使用Sequential序列模型
    model = Sequential()
    # 卷积层
    model.add(Conv2D(32,
                     (3, 3),
                     activation="relu",
                     padding="same",
                     input_shape=(200, 200, 3)))
    # 最大池化层
    model.add(MaxPool2D((2, 2)))  # 池化窗格
    # Flatten层
    model.add(Flatten())
    # 全连接层
    model.add(Dense(128, activation="relu"))  # 128为神经元的个数
    model.add(Dense(1, activation="sigmoid"))
    # 编译模型
    opt = SGD(lr=0.001, momentum=0.9)  # 随机梯度
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_cnn_model():
    # 实例化模型
    model = define_cnn_model()
    # 创建图片生成器
    datagen = ImageDataGenerator(rescale=1. / 255)
    train_it = datagen.flow_from_directory(
        "./train/",
        class_mode="binary",
        batch_size=64,
        target_size=(200, 200))  # batch_size:一次拿出多少张照片 targe_size:将图片缩放到一定比例
    # 训练模型
    model.fit_generator(train_it,
                        steps_per_epoch=len(train_it),
                        epochs=7,
                        verbose=1)
    model.save("my_model7.h5")

train_cnn_model()
