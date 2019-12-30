# coding:utf-8
# -----------------------------------------------------------------------------
# Comments  : 模型训练时进行数据处理的工具文件
# Developer : SWLIU
# Date      : 2019-12-27
# -----------------------------------------------------------------------------
from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    参数为多个函数名，按照reduce的功能执行，把前一个函数的结果作为下一个函数的输入，直到最后执行完毕
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size  # 原始图像是1200x1800
    w, h = size # 转换为416x416
    scale = min(w/iw, h/ih)  # 转换比例
    nw = int(iw*scale)  # 新图像的宽，保证新图像是等比下降的
    nh = int(ih*scale)  # 新图像的高

    image = image.resize((nw,nh), Image.BICUBIC)  # 缩小图像
    new_image = Image.new('RGB', size, (128,128,128))  # 生成灰色图像
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))  # 将图像填充为中间图像，两侧为灰色的样式
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''
    random preprocessing for real-time data augmentation
    获取真实的数据根据输入的尺寸对原始数据进行缩放处理得到input_shape大小的数据图片，
    随机进行图片的翻转，标记数据数据也根据比例改变
        annotation_line： 单条图片的信息的列表
        input_shape：输入的尺寸
    这里的get_random_data就是对原始数据进行处理的函数，获取真实的数据根据输入的尺寸对原始数据进行缩放处理得到input_shape大小
    的数据图片，随机进行图片的翻转，标记数据数据也根据比例改变
    '''
    # 处理图片
    line = annotation_line.split()
    # 读取图片图片
    image = Image.open(line[0])
    # 原始图片的比例
    iw, ih = image.size
    # 获取模型的输入图片的大小
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        # 获取原始图片和模型输入图片的比例
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            # 首先创建一张灰色的图片
            new_image = Image.new('RGB', (w,h), (128,128,128))
            # 把原始的图片粘贴到灰色图片上
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        # 对所有的图片中的目标进行缩放
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes] # 最多只取20个
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    # 随机的图片比例变换
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    # 计算新的图片尺寸
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    # 改变图片尺寸
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    # 随机把图片摆放在灰度图片上
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    # 是否反转图片
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    # 在HSV坐标域中，改变图片的颜色范围，hue值相加，sat和vat相乘，
    # 先由RGB转为HSV，再由HSV转为RGB，添加若干错误判断，避免范围过大
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # correct boxes
    # 将所有的图片变换，增加至检测框中，并且包含若干异常处理，
    # 避免变换之后的值过大或过小，去除异常的box。
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        # 变换所有目标的尺寸
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        # 如果已经翻转了需要进行坐标变换，并且把坐标限制在图片内
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        # 最大的目标数不能超过超参数
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data
