# 导入所需的模块
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import util
from skimage.measure import compare_ssim as ssim


# 定义训练函数
def train(args, model, sess, saver):
    # 如果需要微调，则加载预训练模型
    if args.fine_tuning:
        saver.restore(sess, args.pre_trained_model)
        print("saved model is loaded for fine-tuning!")
        print("model path is %s" % (args.pre_trained_model))

    # 获取训练集中图像的数量
    num_imgs = len(os.listdir(args.train_Sharp_path))

    # 创建一个tf记录器，用于记录训练日志
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs', sess.graph)

    # 如果需要在训练期间进行测试，则打开文件以记录结果
    if args.test_with_train:
        f = open("valid_logs.txt", 'w')

    # 初始化变量
    epoch = 0
    step = num_imgs // args.batch_size

    # 如果需要将图像加载到内存中
    if args.in_memory:

        # 加载模糊和清晰图像
        blur_imgs = util.image_loader(args.train_Blur_path, args.load_X, args.load_Y)
        sharp_imgs = util.image_loader(args.train_Sharp_path, args.load_X, args.load_Y)

        # 训练模型
        while epoch < args.max_epoch:

            # 随机打乱索引
            random_index = np.random.permutation(len(blur_imgs))

            # 遍历所有批次
            for k in range(step):
                s_time = time.time()

                # 生成批次图像数据
                blur_batch, sharp_batch = util.batch_gen(blur_imgs, sharp_imgs, args.patch_size, args.batch_size,
                                                         random_index, k, args.augmentation)

                # 判别器更新
                for t in range(args.critic_updates):
                    _, D_loss = sess.run([model.D_train, model.D_loss],
                                         feed_dict={model.blur: blur_batch, model.sharp: sharp_batch,
                                                    model.epoch: epoch})

                # 生成器更新
                _, G_loss, D_loss_fk, content_loss, G_sqrt_loss, gene_color_loss, gene_derive_loss = sess.run(
                    [model.G_train, model.G_loss, model.D_loss_fk, model.content_loss, model.G_sqrt_loss,
                     model.gene_color_loss, model.gene_derive_loss],
                    feed_dict={model.blur: blur_batch, model.sharp: sharp_batch, model.epoch: epoch})

                e_time = time.time()

            # 如果需要记录训练日志，则执行以下操作
            if epoch % args.log_freq == 0:
                summary = sess.run(merged, feed_dict={model.blur: blur_batch, model.sharp: sharp_batch})
                train_writer.add_summary(summary, epoch)

                # 如果需要在训练期间进行测试，则执行以下操作
                if args.test_with_train:
                    test(args, model, sess, saver, f, epoch, loading=False)


def test(args, model, sess, saver, file, step=-1, loading=False):
    # 进行测试
    if loading:
        # 加载预训练模型
        saver.restore(sess, args.pre_trained_model)
        print("已加载保存的模型进行测试！")
        print("模型路径为 %s" % args.pre_trained_model)

    # 加载待处理的模糊图像和对应的清晰图像
    blur_img_name = sorted(os.listdir(args.test_Blur_path))
    sharp_img_name = sorted(os.listdir(args.test_Sharp_path))

    PSNR_list = []  # 记录所有测试样本的 PSNR
    ssim_list = []  # 记录所有测试样本的 SSIM

    if args.in_memory:
        # 如果全部图像可放入内存，则一次性读入所有图像
        blur_imgs = util.image_loader(args.test_Blur_path, args.load_X, args.load_Y, is_train=False)
        sharp_imgs = util.image_loader(args.test_Sharp_path, args.load_X, args.load_Y, is_train=False)

        # 对每个图像进行测试
        for i, ele in enumerate(blur_imgs):
            blur = np.expand_dims(ele, axis=0)
            sharp = np.expand_dims(sharp_imgs[i], axis=0)
            # 运行模型进行测试
            output, psnr, ssim = sess.run([model.output, model.PSNR, model.ssim],
                                          feed_dict={model.blur: blur, model.sharp: sharp})
            # 保存测试结果
            if args.save_test_result:
                output = Image.fromarray(output[0])
                split_name = blur_img_name[i].split('.')
                output.save(os.path.join(args.result_path, '%s_sharp.png' % (''.join(map(str, split_name[:-1])))))

            PSNR_list.append(psnr)
            ssim_list.append(ssim)
    else:
        # 如果图像过大不能全部放入内存，则使用数据流方式读取
        sess.run(model.data_loader.init_op['val_init'])
        for i in range(len(blur_img_name)):
            # 运行模型进行测试
            output, psnr, ssim = sess.run([model.output, model.PSNR, model.ssim])
            # 保存测试结果
            if args.save_test_result:
                output = Image.fromarray(output[0])
                split_name = blur_img_name[i].split('.')
                output.save(os.path.join(args.result_path, '%s_sharp.png' % (''.join(map(str, split_name[:-1])))))

            PSNR_list.append(psnr)
            ssim_list.append(ssim)

    # 计算所有测试样本的 PSNR 和 SSIM 的平均值
    length = len(PSNR_list)
    mean_PSNR = sum(PSNR_list) / length
    mean_ssim = sum(ssim_list) / length

    if step == -1:
        # 如果没有指定训练步数，则在文件中记录 PSNR 和 SSIM 的平均值
        file.write('PSNR: %.4f, SSIM: %.4f' % (mean_PSNR, mean_ssim))
        file.close()
    else:


# 如果指定了训练步数，则在文件中记录当前步数和 PSNR、SSIM 的平均值
# 定义 test_only 函数，接受参数 args, model, sess, saver
def test_only(args, model, sess, saver):
    # 从预训练模型文件中恢复模型
    saver.restore(sess, args.pre_trained_model)
    print("saved model is loaded for test only!")
    print("model path is %s" % args.pre_trained_model)

    # 读取测试模糊图像的文件名列表
    blur_img_name = sorted(os.listdir(args.test_Blur_path))

    # 如果参数 in_memory 为 True，则将所有测试模糊图像加载到内存中
    if args.in_memory:

        # 调用 image_loader 函数将测试模糊图像加载到内存中
        blur_imgs = util.image_loader(args.test_Blur_path, args.load_X, args.load_Y, is_train=False)

        # 遍历所有测试模糊图像
        for i, ele in enumerate(blur_imgs):

            # 将当前测试模糊图像转换为 numpy 数组，并在第 0 维增加一个维度
            blur = np.expand_dims(ele, axis=0)

            # 如果参数 chop_forward 为 True，则进行切块前向传递
            if args.chop_forward:

                # 调用 recursive_forwarding 函数进行切块前向传递，并将结果转换为 Image 对象
                output = util.recursive_forwarding(blur, args.chop_size, sess, model, args.chop_shave)
                output = Image.fromarray(output[0])

            # 否则，直接进行正常的前向传递
            else:

                # 进行前向传递，并将结果转换为 Image 对象
                output = sess.run(model.output, feed_dict={model.blur: blur})
                output = Image.fromarray(output[0])

            # 将输出结果保存为文件
            split_name = blur_img_name[i].split('.')
            output.save(os.path.join(args.result_path, '%s_sharp.png' % (''.join(map(str, split_name[:-1])))))

    # 否则，每次只加载一个测试模糊图像
    else:

        # 初始化测试数据集的迭代器
        sess.run(model.data_loader.init_op['te_init'])

        # 遍历所有测试模糊图像
        for i in range(len(blur_img_name)):
            # 进行前向传递，并将结果转换为 Image 对象
            output = sess.run(model.output)
            output = Image.fromarray(output[0])

            # 将输出结果保存为文件
            split_name = blur_img_name[i].split('.')
            output.save(os.path.join(args.result_path, '%s_sharp.png' % (''.join(map(str, split_name[:-1])))))

