import tensorflow as tf
import numpy as np
import os
定义一个类
dataloader，该类包含如下方法：

def __init__(self, args):
    # 初始化 dataloader 的参数
    self.mode = args.mode
    self.patch_size = args.patch_size
    self.batch_size = args.batch_size
    self.train_Sharp_path = args.train_Sharp_path
    self.train_Blur_path = args.train_Blur_path
    self.test_Sharp_path = args.test_Sharp_path
    self.test_Blur_path = args.test_Blur_path
    self.test_with_train = args.test_with_train
    self.test_batch = args.test_batch
    self.load_X = args.load_X
    self.load_Y = args.load_Y
    self.augmentation = args.augmentation
    self.channel = args.channel


def build_loader(self):
    # 根据不同的 mode 构建不同的数据集
    if self.mode == 'train':
        # 读取训练集数据路径
        tr_sharp_imgs = sorted(os.listdir(self.train_Sharp_path))
        tr_blur_imgs = sorted(os.listdir(self.train_Blur_path))
        tr_sharp_imgs = [os.path.join(self.train_Sharp_path, ele) for ele in tr_sharp_imgs]
        tr_blur_imgs = [os.path.join(self.train_Blur_path, ele) for ele in tr_blur_imgs]
        train_list = (tr_blur_imgs, tr_sharp_imgs)

        # 构建训练集数据管道
        self.tr_dataset = tf.data.Dataset.from_tensor_slices(train_list)
        self.tr_dataset = self.tr_dataset.map(self._parse, num_parallel_calls=4).prefetch(32)
        self.tr_dataset = self.tr_dataset.map(self._resize, num_parallel_calls=4).prefetch(32)
        self.tr_dataset = self.tr_dataset.map(self._get_patch, num_parallel_calls=4).prefetch(32)
        if self.augmentation:
            self.tr_dataset = self.tr_dataset.map(self._data_augmentation, num_parallel_calls=4).prefetch(32)
        self.tr_dataset = self.tr_dataset.shuffle(32)
        self.tr_dataset = self.tr_dataset.repeat()
        self.tr_dataset = self.tr_dataset.batch(self.batch_size)

        # 如果需要使用训练集中的一部分作为验证集，构建验证集数据管道
        if self.test_with_train:
            val_sharp_imgs = sorted(os.listdir(self.test_Sharp_path))
            val_blur_imgs = sorted(os.listdir(self.test_Blur_path))
            val_sharp_imgs = [os.path.join(self.test_Sharp_path, ele) for ele in val_sharp_imgs]
            val_blur_imgs = [os.path.join(self.test_Blur_path, ele) for ele in val_blur_imgs]
            valid_list = (val_blur_imgs, val_sharp_imgs)

            self.val_dataset = tf.data.Dataset.from_tensor_slices(valid_list)
            self.val_dataset = self.val_dataset.map(self._parse, num_parallel_calls=4).prefetch(32)
            self.val_dataset = self.val_dataset.batch(self.test_batch)

        # 创建数据迭代器
        iterator = tf.data.Iterator.from_structure(self.tr_dataset.output_types, self.tr_dataset.output_shapes)
        self.next_batch = iterator.get_next()
        self.init_op = {}
        self.init_op['tr_init'] = iterator.make_initializer(self.tr_dataset)


# 定义类
class MyClass:

    # 构造函数
    def __init__(self, test_with_train, mode, test_Sharp_path, test_Blur_path, channel, load_Y, load_X):
        self.test_with_train = test_with_train
        self.mode = mode
        self.test_Sharp_path = test_Sharp_path
        self.test_Blur_path = test_Blur_path
        self.channel = channel
        self.load_Y = load_Y
        self.load_X = load_X

        # 判断模式
        if self.test_with_train:
            # 初始化验证集操作
            self.init_op = {'val_init': None}
            self.init_op['val_init'] = iterator.make_initializer(self.val_dataset)
        elif self.mode == 'test':
            # 准备验证集数据
            val_sharp_imgs = sorted(os.listdir(self.test_Sharp_path))
            val_blur_imgs = sorted(os.listdir(self.test_Blur_path))
            val_sharp_imgs = [os.path.join(self.test_Sharp_path, ele) for ele in val_sharp_imgs]
            val_blur_imgs = [os.path.join(self.test_Blur_path, ele) for ele in val_blur_imgs]
            valid_list = (val_blur_imgs, val_sharp_imgs)

            # 创建验证集数据集
            self.val_dataset = tf.data.Dataset.from_tensor_slices(valid_list)
            self.val_dataset = self.val_dataset.map(self._parse, num_parallel_calls=4).prefetch(32)
            self.val_dataset = self.val_dataset.batch(1)

            # 创建验证集迭代器
            iterator = tf.data.Iterator.from_structure(self.val_dataset.output_types, self.val_dataset.output_shapes)
            self.next_batch = iterator.get_next()
            self.init_op = {'val_init': None}
            self.init_op['val_init'] = iterator.make_initializer(self.val_dataset)
        elif self.mode == 'test_only':
            # 准备测试集数据
            blur_imgs = sorted(os.listdir(self.test_Blur_path))
            blur_imgs = [os.path.join(self.test_Blur_path, ele) for ele in blur_imgs]

            # 创建测试集数据集
            self.te_dataset = tf.data.Dataset.from_tensor_slices(blur_imgs)
            self.te_dataset = self.te_dataset.map(self._parse_Blur_only, num_parallel_calls=4).prefetch(32)
            self.te_dataset = self.te_dataset.batch(1)

            # 创建测试集迭代器
            iterator = tf.data.Iterator.from_structure(self.te_dataset.output_types, self.te_dataset.output_shapes)
            self.next_batch = iterator.get_next()
            self.init_op = {'te_init': None}
            self.init_op['te_init'] = iterator.make_initializer(self.te_dataset)

    # 解析数据函数
    def _parse(self, image_blur, image_sharp):

        image_blur = tf.read_file(image_blur)
        image_sharp = tf.read_file(image_sharp)

        image_blur = tf.image.decode_png(image_blur, channels=self.channel)
        image_sharp = tf.image.decode_png(image_sharp, channels=self.channel)

        image_blur = tf.cast(image_blur, tf.float32)
        image_sharp = tf.cast(image_sharp, tf.float32)

        return image_blur, image_sharp

        # 图像缩放函数

    def _parse_Blur_only(self, image_blur):

        image_blur = tf.read_file(image_blur)
        image_blur = tf.image.decode_image(image_blur, channels=self.channel)
        image_blur = tf.cast(image_blur, tf.float32)

        return image_blur

        # 获取图像尺寸
        shape = tf.shape(image_blur)
        ih = shape[0]
        iw = shape[1]

        # 随机裁剪位置
        ix = tf.random_uniform(shape=[1], minval=0, maxval=iw - self.patch_size + 1, dtype=tf.int32)[0]
        iy = tf.random_uniform(shape=[1], minval=0, maxval=ih - self.patch_size + 1, dtype=tf.int32)[0]

        # 裁剪图像块
        img_sharp_in = image_sharp[iy:iy + self.patch_size, ix:ix + self.patch_size]
        img_blur_in = image_blur[iy:iy + self.patch_size, ix:ix + self.patch_size]

        return img_blur_in, img_sharp_in


# 进行数据增强
def _data_augmentation(self, image_blur, image_sharp):
    # 随机旋转、翻转
    rot = tf.random_uniform(shape=[1], minval=0, maxval=3, dtype=tf.int32)[0]
    flip_rl = tf.random_uniform(shape=[1], minval=0, maxval=3, dtype=tf.int32)[0]
    flip_updown = tf.random_uniform(shape=[1], minval=0, maxval=3, dtype=tf.int32)[0]

    # 旋转图像
    image_blur = tf.image.rot90(image_blur, rot)
    image_sharp = tf.image.rot90(image_sharp, rot)

    # 翻转图像
    rl = tf.equal(tf.mod(flip_rl, 2), 0)
    ud = tf.equal(tf.mod(flip_updown, 2), 0)

    image_blur = tf.cond(rl, true_fn=lambda: tf.image.flip_left_right(image_blur), false_fn=lambda: (image_blur))
    image_sharp = tf.cond(rl, true_fn=lambda: tf.image.flip_left_right(image_sharp), false_fn=lambda: (image_sharp))

    image_blur = tf.cond(ud, true_fn=lambda: tf.image.flip_up_down(image_blur), false_fn=lambda: (image_blur))
    image_sharp = tf.cond(ud, true_fn=lambda: tf.image.flip_up_down(image_sharp), false_fn=lambda: (image_sharp))

    return image_blur, image_sharp
