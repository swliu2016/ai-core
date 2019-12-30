# coding:utf-8
# -----------------------------------------------------------------------------
# Comments  : 把xml文件转换为txt
# Developer : SWLIU
# Date      : 2019-12-27
# -----------------------------------------------------------------------------
import xml.etree.ElementTree as ET
import os
from config import cfg

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = cfg.ROOT.CLASSES

rootPath = cfg.ROOT.PATH

def convert_annotation(image_id, list_file):
    in_file = open('%s/Annotations/%s.xml'%(rootPath, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = os.getcwd()

for year, image_set in sets:
    image_ids = open('%s/ImageSets/Main/%s.txt'%(rootPath, image_set)).read().strip().split()
    list_file = open('%s/%s.txt'%(rootPath, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/JPEGImages/%s.jpg'%(rootPath, image_id))
        convert_annotation(image_id, list_file)
        list_file.write('\n')
    list_file.close()

