# coding:utf-8
# -----------------------------------------------------------------------------
# Comments  : 批量修改文件名称
# Developer : SWLIU
# Date      : 2019-12-23
# -----------------------------------------------------------------------------
import os

def addSuffixName(rootPath, oldFileName, newFileNameSuffix):
    """
    给文件加后缀
    :param rootPath:
    :param oldFileName:
    :param newFileNameSuffix:
    :return:
    """
    oldFilePath = rootPath + oldFileName
    new_name = rootPath + newFileNameSuffix + "_" + oldFileName
    os.rename(oldFilePath, new_name)
    print('新增后缀成功：' + new_name)

if __name__ =='__main__':
    path = "data_set/VOCdevkit-smoke/VOCdevkit/VOC2007/Annotations/"
    filename_list = os.listdir(path)
    a = 0
    for i in filename_list:
        used_name = path + filename_list[a]
        addSuffixName(path, filename_list[a], str(a))
        a += 1
    print('修改完成')