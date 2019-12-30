# coding:utf-8
# -----------------------------------------------------------------------------
# Comments  : 输入得到的txt文件，通过聚类得到数据最佳anchors。
# Developer : SWLIU
# Date      : 2019-12-27
# -----------------------------------------------------------------------------
import numpy as np
import os
from config import cfg

'''
k-means拿到数据里的所有目标框，得到所有的宽和高，在这里随机取得9个随机中心，之后以9个点为中心得到9个族，不断计算其他点到中点的
距离，调整每个点所属的族和中心，直到9个中心不变即可，这9个中心的x,y就是整个数据的9个合适的anchors==框的宽和高
'''
class YOLO_Kmeans:
    def __init__(self, cluster_number, filename):
        # 读取kmeans的中心数
        self.cluster_number = cluster_number
        # 标签文件的文件名
        self.filename = filename

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        # clusters:9个随机中心点【width,height】
        n = boxes.shape[0]
        k = self.cluster_number

        # 所有的boxex的面积
        box_area = boxes[:, 0] * boxes[:, 1]
        # 将box_area的每个元素重复k次
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))
        # 计算9个中点的面积
        cluster_area = clusters[:, 0] * clusters[:, 1]
        # 对cluster_area进行复制N份
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))
        # 获取box和中心点的交叉w的宽
        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)
        # 获取box和中心点的交叉w的高
        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        # 交叉点的面积
        inter_area = np.multiply(min_w_matrix, min_h_matrix)
        # 9个交叉点和所有的boxes的iou值
        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        # 计算9个中心点与所有boxes总的iou,n个点的平均iou
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        # np.median求众数
        # boxes = [宽，高]C
        # K 中心点数
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        # 从所有的boxe中选取9个随机中心点
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:
            # 计算所有的boxes和clusters的值（n,k)
            distances = 1 - self.iou(boxes, clusters)
            # 选取iou值最小的点（n,）
            current_nearest = np.argmin(distances, axis=1)
            # 中心点未改变，跳出
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            # 计算每个群组的中心或者众数
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)
            # 改变中心点
            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        # 把九个中心点，写入txt文件
        f = open(os.path.join(cfg.ROOT.PATH, "Model/yolo_anchors.txt"), 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        # 打开文件
        f = open(self.filename, 'r')
        dataSet = []
        # 读取文件
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            # infons[0] 为图片的名称
            for i in range(1, length):
                #获取文件的宽和高
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result

    def txt2clusters(self):
        # 获取所有目标的宽和高，width,height
        all_boxes = self.txt2boxes()
        # result9个中心点
        result = self.kmeans(all_boxes, k=self.cluster_number)
        # 按最后一列顺序排序
        result = result[np.lexsort(result.T[0, None])]
        # 把结果写到txt文件
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        # 计算9个中心点与所有的boxes总的iou,n个点的平均iou
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))

if __name__ == "__main__":
    cluster_number = 9
    filename = os.path.join(cfg.ROOT.PATH, "train.txt")
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters()
