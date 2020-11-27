# * coding: utf8 *
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
# 用于加速运算
from numba import jit

'''
EigenFace:利用PCA求得特征脸
FisherFace:利用PCA+LDA求得特征脸
'''
SINGLE_CLASS_SAMPLES_NUM = 11
# 类别数
CLASS_NUM = 15
classes_info = []
# Adjustable
TRAIN_SET_RATIO = 0.7
# Adjustable
EIGEN_FACES_NUM = 14

np.set_printoptions(threshold=np.inf)


# 日志函数
def logger(msg):
    print(msg)


# 存储类别信息（类别序号、类别名称、包含图片的路径）
class ClassInfo:
    def __init__(self, label_index, label_name, images_path):
        self.label_index = label_index
        self.label_name = label_name
        self.images_path = images_path

    def show_info(self):
        print(self.label_index)
        print(self.label_name)
        print(self.images_path)


# 读取数据集信息，存储在classInfo对象中
def load_dataset_info(path):
    _classes_info = []
    k = 0
    class_names = os.listdir(path)
    for class_name in class_names:
        if class_name == '_exclude':
            continue
        img_names = os.listdir(os.path.join(path, class_name))
        img_paths = [os.path.join(path, class_name, img_name) for img_name in img_names]
        _classes_info.append(ClassInfo(k, class_name, img_paths))
        k = k + 1
    return len(_classes_info), _classes_info


# 读取指定类别的指定图片
@jit
def get_image(label_index, img_index):
    # 图像灰度化处理
    img = cv2.imread(classes_info[label_index].images_path[img_index], cv2.IMREAD_GRAYSCALE)
    # 图像标准化
    return img


# 划分构建70%训练集和30%测试集
@jit
def build_dataset():
    # 从每个类别中随机选取训练集和测试集
    training_samples_num = int(np.round(SINGLE_CLASS_SAMPLES_NUM * TRAIN_SET_RATIO))
    # 训练集、测试集
    training_data, testing_data, training_label, testing_label = [], [], [], []
    for i in range(CLASS_NUM):
        whole_img_index = np.arange(0, SINGLE_CLASS_SAMPLES_NUM)
        training_img_index = np.random.choice(np.arange(SINGLE_CLASS_SAMPLES_NUM), size=training_samples_num,
                                              replace=False)
        for img_index in whole_img_index:
            if img_index in training_img_index:
                training_data.append(get_image(i, img_index))
                training_label.append(i)
            else:
                testing_data.append(get_image(i, img_index))
                testing_label.append(i)
        logger('Finish building Dataset of Class {0}-{1}.'.format(i, classes_info[i].label_name))
    return training_data, training_label, testing_data, testing_label


# 将训练集图片转换为行向量
def as_row_matrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((0, X[0].size), dtype=X[0].dtype)
    for row in X:
        mat = np.vstack((mat, np.asarray(row).reshape(1, -1)))
    return mat


# 主成分分析
def pca(X, y, num_components=0):
    [n, d] = X.shape
    if (num_components <= 0) or (num_components > n):
        num_components = n
    mean_face = X.mean(axis=0)
    X = X - mean_face
    if n > d:
        C = np.dot(X.T, X)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X, X.T)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T, eigenvectors)
        for i in range(n):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])

    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvalues = eigenvalues[0:num_components].copy()
    eigenvectors = eigenvectors[:, 0:num_components].copy()
    return [eigenvalues, eigenvectors, mean_face]


# lda
def lda(X, y, num_components=0):
    y = np.asarray(y)
    [n, d] = X.shape
    c = np.unique(y)
    if (num_components <= 0) or (num_components > (len(c) - 1)):
        num_components = (len(c) - 1)
    meanTotal = X.mean(axis=0)
    Sw = np.zeros((d, d), dtype=np.float32)
    Sb = np.zeros((d, d), dtype=np.float32)
    for i in c:
        Xi = X[np.where(y == i)[0], :]
        meanClass = Xi.mean(axis=0)
        Sw = Sw + np.dot((Xi - meanClass).T, (Xi - meanClass))
        Sb = Sb + n * np.dot((meanClass - meanTotal).T, (meanClass - meanTotal))
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw) * Sb)
    idx = np.argsort(-eigenvalues.real)
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
    eigenvalues = np.array(eigenvalues[0:num_components].real, dtype=np.float32, copy=True)
    eigenvectors = np.array(eigenvectors[0:, 0:num_components].real, dtype=np.float32, copy=True)
    return [eigenvalues, eigenvectors]


# 计算image在特征脸上的投影，即image有所有eigen face加权得到
def project(eigen_faces, image, mean_face=None):
    if mean_face is None:
        return np.dot(image, eigen_faces)
    return np.dot(image - mean_face, eigen_faces)


# 构建fisher faces
def fisherfaces(X, y, num_components=0):
    y = np.asarray(y)
    # print X.shape
    [n, d] = X.shape
    c = len(np.unique(y))
    [eigenvalues_pca, eigenvectors_pca, mu_pca] = pca(X, y, (n - c))
    [eigenvalues_lda, eigenvectors_lda] = lda(project(eigenvectors_pca, X, mu_pca), y, num_components)
    eigenvectors = np.dot(eigenvectors_pca, eigenvectors_lda)
    return [eigenvalues_lda, eigenvectors, mu_pca]


# 构建eigen faces
def eigenfaces(X, y):
    y = np.asarray(y)
    # print X.shape
    [n, d] = X.shape
    c = len(np.unique(y))
    [eigenvalues_pca, eigenvectors_pca, mu_pca] = pca(X, y, n - c)
    return [eigenvalues_pca, eigenvectors_pca, mu_pca]


# 返回特征脸
# W:
def facesModel(X, y, num_components):
    [D, W, mu] = fisherfaces(as_row_matrix(X), y, num_components)
    # [D, W, mu] = eigenfaces(as_row_matrix(X), y)
    return W, mu


# 距离函数
def dist_metric(p, q):
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    return np.sqrt(np.sum(np.power((p - q), 2)))


# predict
def predict(eigen_faces, mean_face, projections, y, X):
    minDist = np.finfo('float').max
    # print minDist
    minClass = -1
    # 计算unknown face在特征脸上的投影
    Q = project(eigen_faces, X.reshape(1, -1), mean_face)
    for i in range(len(projections)):
        dist = dist_metric(projections[i], Q)
        if dist < minDist:
            minDist = dist
            minClass = y[i]
    return minClass


# 模型测试
def model_test(testing_data, testing_label):
    hit_cnt = 0
    for i in range(len(testing_label)):
        predict_id = predict(eigen_faces, mean_faces, projections, train_label, testing_data[i])
        if predict_id == testing_label[i]:
            hit_cnt += 1
    rate = hit_cnt / len(testing_label)
    return rate


if __name__ == '__main__':
    # 读取类别信息
    CLASS_NUM, classes_info = load_dataset_info('./YALE_IMAGES')
    # 划分数据集
    train_data, train_label, test_data, test_label = build_dataset()
    # 构建fisher faces模型
    eigen_faces, mean_faces = facesModel(train_data, train_label, EIGEN_FACES_NUM)
    # 训练集投影集合
    projections = []
    for img in train_data:
        # 计算image在特征脸上的投影
        projections.append(project(eigen_faces, img.reshape(1, -1), mean_faces))
    accuracy = model_test(test_data, test_label)
    print('The accuracy is %.2f%%' % (accuracy * 100))
