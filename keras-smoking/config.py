# coding:utf-8
# -----------------------------------------------------------------------------
# Comments  : 创建yolo3所需的train.txt,val.txt,test.txt
# Developer : SWLIU
# Date      : 2019-12-26
# -----------------------------------------------------------------------------

from easydict import EasyDict as edict

__C = edict()
cfg = __C
__C.ROOT = edict()
__C.ROOT.PATH = 'train/smoking'
__C.ROOT.VAL_SPLIT = 0.1
__C.ROOT.BATCH_SIZE = 6
__C.ROOT.EPOCHS = 5000
__C.ROOT.INITIAL_EPOCHS = 0
__C.ROOT.MODEL_RSLT_NAME = "smoking.h5"
__C.ROOT.INPUT_SHAPE = (416, 416)
__C.ROOT.CLASSES = ["smoking"]
__C.ROOT.TRAIN_VAL_PERCENT = 0
__C.ROOT.TRAIN_PERCENT = 0.9
__C.ROOT.PRE_TRAIN_MODEL ="yolov3.h5"





