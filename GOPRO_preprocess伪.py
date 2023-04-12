# 导入必要的库
import os
import subprocess
import argparse
import numpy as np

# 定义命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--GOPRO_path", type=str, default='./GOPRO')
parser.add_argument("--output_path", type=str, default='./output')
parser.add_argument("--train_num", type=int, default=1000)
parser.add_argument("--test_num", type=int, default=10)
parser.add_argument("--is_gamma", type=str2bool, default=False)


# 定义函数：将字符串类型的参数值转化为布尔型
def str2bool(v):
    return v.lower() in ('true')


# 解析命令行参数
args = parser.parse_args()

# 如果训练集的路径不存在，创建目录
if not os.path.exists(os.path.join(args.output_path, 'train')):
    os.mkdir(os.path.join(args.output_path, 'train'))
    os.mkdir(os.path.join(args.output_path, 'train/sharp'))
    os.mkdir(os.path.join(args.output_path, 'train/blur'))

# 如果测试集的路径不存在，创建目录
if not os.path.exists(os.path.join(args.output_path, 'test')):
    os.mkdir(os.path.join(args.output_path, 'test'))
    os.mkdir(os.path.join(args.output_path, 'test/sharp'))
    os.mkdir(os.path.join(args.output_path, 'test/blur'))

# 组装训练集和测试集的路径
GOPRO_train_path = os.path.join(args.GOPRO_path, 'train')
GOPRO_test_path = os.path.join(args.GOPRO_path, 'test')

# 创建空的训练集
train_blur = []
train_sharp = []

# 遍历训练集中每个目录
for direc in sorted(os.listdir(GOPRO_train_path)):
    # 如果使用 gamma 校正，组装模糊图像的路径
    if args.is_gamma:
        blur = os.path.join(os.path.join(GOPRO_train_path, direc), 'blur_gamma')
    # 如果不使用 gamma 校正，组装模糊图像的路径
    else:
        blur = os.path.join(os.path.join(GOPRO_train_path, direc), 'blur')
    # 组装清晰图像的路径
    sharp = os.path.join(os.path.join(GOPRO_train_path, direc), 'sharp')

    # 获取清晰图像列表，并遍历模糊图像列表
    sharp_imgs = sorted(os.listdir(sharp))
    for i, img in enumerate(sorted(os.listdir(blur))):
        # 将模糊图像路径和清晰图像路径添加到训练集中
        train_blur.append(os.path.join(blur, img))
        train_sharp.append(os.path.join(sharp, sharp_imgs[i]))

# 将训练集转化为 numpy 数组，并随机选择指定数量的样本
train_blur = np.asarray(train_blur)
train_sharp = np.asarray(train_sharp)
random_index = np.random.permutation(len(train_blur))[:args.train_num]
# 定义参数
parser = argparse.ArgumentParser()
parser.add_argument("--GOPRO_path", type=str, default='./GOPRO')
parser.add_argument("--output_path", type=str, default='./output')
parser.add_argument("--train_num", type=int, default=1000)
parser.add_argument("--test_num", type=int, default=10)
parser.add_argument("--is_gamma", type=str2bool, default=False)

# 将命令行参数解析为命名空间对象
args = parser.parse_args()

# 如果输出路径下的训练目录不存在，则创建目录
if not os.path.exists(os.path.join(args.output_path, 'train')):
    os.mkdir(os.path.join(args.output_path, 'train'))
    os.mkdir(os.path.join(args.output_path, 'train/sharp'))
    os.mkdir(os.path.join(args.output_path, 'train/blur'))

# 如果输出路径下的测试目录不存在，则创建目录
if not os.path.exists(os.path.join(args.output_path, 'test')):
    os.mkdir(os.path.join(args.output_path, 'test'))
    os.mkdir(os.path.join(args.output_path, 'test/sharp'))
    os.mkdir(os.path.join(args.output_path, 'test/blur'))

# 定义GOPRO数据集的训练和测试路径
GOPRO_train_path = os.path.join(args.GOPRO_path, 'train')
GOPRO_test_path = os.path.join(args.GOPRO_path, 'test')

# 获取训练集的模糊和清晰图像路径
train_blur = []
train_sharp = []

for direc in sorted(os.listdir(GOPRO_train_path)):
    if args.is_gamma:
        blur = os.path.join(os.path.join(GOPRO_train_path, direc), 'blur_gamma')
    else:
        blur = os.path.join(os.path.join(GOPRO_train_path, direc), 'blur')
    sharp = os.path.join(os.path.join(GOPRO_train_path, direc), 'sharp')

    sharp_imgs = sorted(os.listdir(sharp))
    for i, img in enumerate(sorted(os.listdir(blur))):
        train_blur.append(os.path.join(blur, img))
        train_sharp.append(os.path.join(sharp, sharp_imgs[i]))

# 将训练集的模糊和清晰图像路径转换为NumPy数组
train_blur = np.asarray(train_blur)
train_sharp = np.asarray(train_sharp)

# 随机选择指定数量的训练图像
random_index = np.random.permutation(len(train_blur))[:args.train_num]

# 将选定的训练图像复制到输出路径的训练目录下
for index in random_index:
    subprocess.call(['cp', train_blur[index],
                     os.path.join(args.output_path, 'train/blur/%s' % ('_'.join(train_blur[index].split('/')[-3:])))])
    subprocess.call(['cp', train_sharp[index],
                     os.path.join(args.output_path, 'train/sharp/%s' % ('_'.join(train_sharp[index].split('/')[-3:])))])

# 获取测试集的模糊和清晰图像路径
test_blur = []
test_sharp = []

for direc in sorted(os.listdir(GOPRO_test_path)):
    if args.is_gamma:
        blur = os.path.join(os.path.join(GOPRO_test_path, direc）
        else:
        blur = os.path.join(os.path.join(GOPRO_test_path, direc), 'blur')
        sharp = os.path.join(os.path.join(GOPRO_test_path, direc), 'sharp')

        sharp_imgs = sorted(os.listdir(sharp))
        for i, img in enumerate(sorted(os.listdir(blur))):
            test_blur.append(os.path.join(blur, img))
        test_sharp.append(os.path.join(sharp, sharp_imgs[i]))

        test_blur = np.asarray(test_blur)
        test_sharp = np.asarray(test_sharp)
        random_index = np.random.permutation(len(test_blur))[:args.test_num]

        for index in random_index:
            subprocess.call(['cp', test_blur[index], os.path.join(args.output_path, 'test/blur/%s' % (
                '_'.join(test_blur[index].split('/')[-3:])))])
        subprocess.call(['cp', test_sharp[index], os.path.join(args.output_path, 'test/sharp/%s' % (
            '_'.join(test_sharp[index].split('/')[-3:])))])