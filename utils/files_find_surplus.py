# coding:utf-8
# -----------------------------------------------------------------------------
# Comments  : 找到两个文件夹下不同的文件名的文件清单
# Developer : SWLIU
# Date      : 2019-12-24
# -----------------------------------------------------------------------------
import os

objPath = "train/smoking/Annotations"
comparedObjPath= "train/smoking/JPEGImages"
objPath_files = os.listdir(objPath)
comparedObjPath_files = os.listdir(comparedObjPath)

print("检测对象有文件个数："+ str(len(objPath_files)))
print("检测对象有文件个数："+ str(len(comparedObjPath_files)))

for obj in objPath_files:
    objName = obj[:-4]
    flag = False
    for comparedObj in comparedObjPath_files:
        comparedObjName = comparedObj[:-4]
        if objName == comparedObjName:
            flag = True
            break;
    if flag is False:
        print("在对比文件件里未找到对象："+ obj)

print("检测结束。。。。")
