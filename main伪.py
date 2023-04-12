#引入所需要的库
import tensorflow as tf
from Deblur_Net import Deblur_Net
from mode import *
import argparse
import os
parser = argparse.ArgumentParser()

定义函数 str2bool(v)
    返回 v.lower() 是否等于 'true'

定义参数解析器 parser
解析器添加参数:
    --channel (类型为 int, 默认值为 3)
    --n_feats (类型为 int, 默认值为 64)
    --num_of_down_scale (类型为 int, 默认值为 2)
    --gen_resblocks (类型为 int, 默认值为 9)
    --discrim_blocks (类型为 int, 默认值为 3)
    --train_Sharp_path (类型为 str, 默认值为 "./sharp/")
    --train_Blur_path (类型为 str, 默认值为 "./blur")
    --test_Sharp_path (类型为 str, 默认值为 "./val_sharp")
    --test_Blur_path (类型为 str, 默认值为 "./val_blur")
    --vgg_path (类型为 str, 默认值为 "./vgg19/vgg19.npy")
    --patch_size (类型为 int, 默认值为 256)
    --result_path (类型为 str, 默认值为 "./result")
    --model_path (类型为 str, 默认值为 "./model")
    --in_memory (类型为 bool, 默认值为 True)
    --batch_size (类型为 int, 默认值为 1)
    --max_epoch (类型为 int, 默认值为 300)
    --learning_rate (类型为 float, 默认值为 1e-4)
    --decay_step (类型为 int, 默认值为 150)
    --test_with_train (类型为 bool, 默认值为 True)
    --save_test_result (类型为 bool, 默认值为 False)
    --mode (类型为 str, 默认值为 "train")
    --critic_updates (类型为 int, 默认值为 5)
    --augmentation (类型为 bool, 默认值为 False)
    --load_X (类型为 int, 默认值为 640)
    --load_Y (类型为 int, 默认值为 360)
    --fine_tuning (类型为 bool, 默认值为 False)
    --log_freq (类型为 int, 默认值为 1)
    --model_save_freq (类型为 int, 默认值为 50)
    --test_batch (类型为 int, 默认值为 1)
    --pre_trained_model (类型为 str, 默认值为 "./")
    --chop_forward (类型为 bool, 默认值为 False)
    --chop_size (类型为 int, 默认值为 80000)
    --chop_shave (类型为 int, 默认值为 16)

使用参数解析器解析参数 args

定义模型 model，使用 args 作为参数构建模型
调用模型的 build_graph() 方法

输出 "Build model!"

设置环境变量 'CUDA_VISIBLE_DEVICES' 为 '2'

定义 TensorFlow 配置 config，设置 GPU 内存按需分配
创建 TensorFlow 会话 sess，使用 config 作为参数
运行 sess，初始化全局变量
创建 Saver 对象 saver，用于保存训练好的模型

如果 args.mode == 'train'，则
    调用 train(args, model, sess, saver) 函数
否则如果 args.mode == 'test'，则
    打开文件 "test_results.txt"，写入模式 'w'，将其赋值给变量 f
    调用 test(args, model, sess, saver,）
    

