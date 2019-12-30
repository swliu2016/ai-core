# coding:utf-8
# -----------------------------------------------------------------------------
# Comments  : 进行yolov3训练的文件(带预训练权重)
# Developer : SWLIU
# Date      : 2019-12-27
# -----------------------------------------------------------------------------

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import os
import sys
sys.path.append('..')
from yolov3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolov3.utils import get_random_data
from config import cfg

def _main():
    annotation_path = os.path.join(cfg.ROOT.PATH, 'train.txt')
    log_dir = os.path.join(cfg.ROOT.PATH, 'logs/000/')
    classes_path = os.path.join(cfg.ROOT.PATH, 'Model/voc_classes.txt')
    anchors_path = os.path.join(cfg.ROOT.PATH, 'Model/yolo_anchors.txt')
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = cfg.ROOT.INPUT_SHAPE  # multiple of 32, hw

    model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=os.path.join(cfg.ROOT.PATH, cfg.ROOT.PRE_TRAIN_MODEL)) # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    # reduce_lr：当评价指标不在提升时，减少学习率，每次减少10%，当验证损失值，持续3次未减少时，则终止训练。
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

    # early_stopping：当验证集损失值，连续增加小于0时，持续10个epoch，则终止训练。
    # monitor：监控数据的类型，支持acc、val_acc、loss、val_loss等；
    # min_delta：停止阈值，与mode参数配合，支持增加或下降；
    # mode：min是最少，max是最多，auto是自动，与min_delta配合；
    # patience：达到阈值之后，能够容忍的epoch数，避免停止在抖动中；
    # verbose：日志的繁杂程度，值越大，输出的信息越多。
    # min_delta和patience需要相互配合，避免模型停止在抖动的过程中。min_delta降低，patience减少；而min_delta增加，
    # 则patience增加。
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    # 训练和验证的比例
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    '''
    把目标当成一个输入，构成多输入模型，把loss写成一个层，作为最后的输出，搭建模型的时候，
    就只需要将模型的output定义为loss，而compile的时候，
    直接将loss设置为y_pred（因为模型的输出就是loss，所以y_pred就是loss），
    无视y_true，训练的时候，y_true随便扔一个符合形状的数组进去就行了。    
    '''
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 16
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        # 存储最终的参数，再训练过程中，通过回调存储
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    # 全部训练
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 32 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        '''
        在训练中，模型调用fit_generator方法，按批次创建数据，输入模型，进行训练。其中，数据生成器wrapper是data_generator_
        wrapper，用于验证数据格式，最终调用data_generator
        annotation_lines：标注数据的行，每行数据包含图片路径，和框的位置信息；
        batch_size：批次数，每批生成的数据个数；
        input_shape：图像输入尺寸，如(416, 416)；
        anchors：anchor box列表，9个宽高值；
        num_classes：类别的数量；
        '''
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(os.path.join(cfg.ROOT.PATH, cfg.ROOT.MODEL_RSLT_NAME))

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path=os.path.join(cfg.ROOT.PATH, cfg.ROOT.PRE_TRAIN_MODEL)):
    '''
    create the training model
    input_shape：输入图片的尺寸，默认是(416, 416)；
    anchors：默认的9种anchor box，结构是(9, 2)；
    num_classes：类别个数，在创建网络时，只需类别数即可。在网络中，类别值按0~n排列，同时，输入数据的类别也是用索引表示；
    load_pretrained：是否使用预训练权重。预训练权重，既可以产生更好的效果，也可以加快模型的训练速度；
    freeze_body：冻结模式，1或2。其中，1是冻结DarkNet53网络中的层，2是只保留最后3个1x1的卷积层，其余层全部冻结；
    weights_path：预训练权重的读取路径；
    '''
    K.clear_session()  # 清除session
    h, w = input_shape  # 尺寸
    image_input = Input(shape=(w, h, 3))  # 图片输入格式
    num_anchors = len(anchors)  # anchor数量

    # YOLO的三种尺度，每个尺度的anchor数，类别数+边框4个+置信度1
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    # 加载预训练模型
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    # 构建 yolo_loss
    # model_body: [(?, 13, 13, 18), (?, 26, 26, 18), (?, 52, 52, 18)]
    # y_true: [(?, 13, 13, 18), (?, 26, 26, 18), (?, 52, 52, 18)]
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''
    data generator for fit_generator
    annotation_lines: 所有的图片名称
    batch_size：每批图片的大小
    input_shape： 图片的输入尺寸
    anchors: 大小
    num_classes： 类别数
    '''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                # 随机排列图片顺序
                np.random.shuffle(annotation_lines)
            # image_data: (16, 416, 416, 3)
            # box_data: (16, 20, 5) # 每个图片最多含有20个框
            # 获取图片和盒子
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            # 获取真实的数据根据输入的尺寸对原始数据进行缩放处理得到input_shape大小的数据图片，
            # 随机进行图片的翻转，标记数据数据也根据比例改变
            # 添加图片
            image_data.append(image)
            # 添加盒子
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        # y_true是3个预测特征的列表
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        # y_true的第0和1位是中心点xy，范围是(0~13/26/52)，第2和3位是宽高wh，范围是0~1，
        # 第4位是置信度1或0，第5~n位是类别为1其余为0。
        # [(16, 13, 13, 3, 6), (16, 26, 26, 3, 6), (16, 52, 52, 3, 6)]
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """
    用于条件检查
    """
    # 标注图片的行数
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()
