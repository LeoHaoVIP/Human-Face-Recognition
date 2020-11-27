# * coding: utf8 *
import numpy as np
import cv2
import os
import time
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
# 快速计算距离矩阵
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
# 用于加速运算
from numba import jit

# 单个类别下的样本数目
SINGLE_CLASS_SAMPLES_NUM = 11
# 类别数
CLASS_NUM = 15
# Adjustable
LEARNING_SAMPLES_NUM = 11
# Self-Adjustable
CLUSTER_SIZE = 50
# Adjustable
BINS_NUM = CLASS_NUM * CLUSTER_SIZE
# Adjustable
TRAIN_SET_RATIO = 0.6
classes_info = []
dictionary = []
# Adjustable 描述子类型 [SIFT, SURF, ORB]
DESCRIPTOR_TYPE = 'SIFT'


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


# 加载用于视觉词典生成的样本
@jit
def load_learning_images(label_index):
    images = []
    # 从每个类别中随机选取 LEARNING_SAMPLES_NUM 个样本
    imgs_index = np.random.choice(np.arange(SINGLE_CLASS_SAMPLES_NUM), size=LEARNING_SAMPLES_NUM, replace=False)
    # read image
    for img_index in imgs_index:
        images.append(get_image(label_index, img_index))
    return images


def print_keypoints(keypoints):
    for kp in keypoints:
        print(kp.pt)


# 获取图像描述子
def img2descriptors(image):
    # 创建描述子提取引擎
    if DESCRIPTOR_TYPE == 'SIFT':
        des_engine = cv2.xfeatures2d.SIFT_create()
    elif DESCRIPTOR_TYPE == 'SURF':
        des_engine = cv2.xfeatures2d.SURF_create()
    elif DESCRIPTOR_TYPE == 'ORB':
        des_engine = cv2.ORB_create()
    else:
        des_engine = None
    # 获取图像描述子（特征向量）
    _, descriptors = des_engine.detectAndCompute(image, None)
    cv2.drawKeypoints(image, _, image, (0, 0, 0))
    return descriptors


# 词典学习(return 聚类中心组成的视觉词典)
def build_dictionary():
    global CLUSTER_SIZE
    # 存储每个类别的词典
    _dictionary = None
    # 加载并构建每类图片的视觉词典
    for i in range(CLASS_NUM):
        # start = time.time()
        # 加载词典构建所需图片集
        learning_images = load_learning_images(i)
        # 当前类别图片的descriptors
        descriptors = None
        for image in learning_images:
            # 当前图像的描述子
            img_descriptors = img2descriptors(image)
            descriptors = img_descriptors if descriptors is None else np.vstack((descriptors, img_descriptors))
        # 设置聚类大小为每张图片关键点的平均数量
        CLUSTER_SIZE = np.int(len(descriptors) / LEARNING_SAMPLES_NUM)
        k_means = KMeans(n_clusters=CLUSTER_SIZE, n_init=10)
        k_means.fit(descriptors)
        _dictionary = k_means.cluster_centers_ if _dictionary is None else np.vstack(
            (_dictionary, k_means.cluster_centers_))
        # end = time.time()
        # print('time cost ', end - start)
        logger('Finish building Dictionary of Class {0}-{1}.'.format(i, classes_info[i].label_name))
    return _dictionary


# 获取给定图像的histogram
def img2histogram(image):
    global BINS_NUM
    # start = time.time()
    # 获取图像描述子
    descriptors = img2descriptors(image)
    # 计算每个pixel的descriptor到每个word的欧氏距离
    # SciPy的c-dist计算距离矩阵的速度更快
    distances = cdist(descriptors, dictionary)
    # dictionary中的word在各个像素点上的映射
    word_map = np.argmin(distances, 1)
    # 构造直方图，大小为BINS_NUM
    BINS_NUM = CLASS_NUM * CLUSTER_SIZE
    [histogram, _] = np.histogram(word_map, BINS_NUM, range=(0, BINS_NUM - 1))
    # IMPORTANT：归一化是必要的，因为每张图片的特征点数量不同
    if np.sum(histogram) > 0:
        histogram = histogram / np.sum(histogram)
    return histogram


# 划分构建70%训练集和30%测试集
@jit
def build_dataset():
    # 从每个类别中随机选取训练集和测试集
    training_samples_num = int(np.round(SINGLE_CLASS_SAMPLES_NUM * TRAIN_SET_RATIO))
    # 训练集、测试集数据
    training_data, testing_data = None, None
    # 训练集、测试集标签
    training_label, testing_label = [], []
    for i in range(CLASS_NUM):
        whole_img_index = np.arange(0, SINGLE_CLASS_SAMPLES_NUM)
        training_img_index = np.random.choice(np.arange(SINGLE_CLASS_SAMPLES_NUM), size=training_samples_num,
                                              replace=False)
        for img_index in whole_img_index:
            # 当前图像的直方图
            histogram = img2histogram(get_image(i, img_index))
            if img_index in training_img_index:
                training_data = histogram if training_data is None else np.vstack((training_data, histogram))
                training_label.append(i)
            else:
                testing_data = histogram if testing_data is None else np.vstack((testing_data, histogram))
                testing_label.append(i)
        logger('Finish building Dataset of Class {0}-{1}.'.format(i, classes_info[i].label_name))
    return training_data, training_label, testing_data, testing_label


# 加载本地数据集
def read_from_local_dataset(path):
    training_data = np.loadtxt(path + '/train_data')
    training_label = np.loadtxt(path + '/train_label')
    testing_data = np.loadtxt(path + '/test_data')
    testing_label = np.loadtxt(path + '/test_label')
    return training_data, training_label, testing_data, testing_label


# 将划分的数据集写入本地
def write_to_local_dataset(path, training_data, training_label, testing_data, testing_label):
    np.savetxt(path + '/train_data', training_data)
    np.savetxt(path + '/train_label', training_label)
    np.savetxt(path + '/test_data', testing_data)
    np.savetxt(path + '/test_label', testing_label)


# histogram距离函数（卡方距离）（论文中的chi^2）
def chi_square_dist(h1, h2):
    eps = 1e-10
    distance = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(h1, h2)])
    return distance


# KNN分类模型训练
def train_model_knn(training_data, training_label):
    # 设置K=1，且根据论文，距离衡量函数为卡方距离-chi_square
    _knn = KNeighborsClassifier(n_neighbors=1, metric=chi_square_dist)
    _knn.fit(np.array(training_data, dtype=np.float32), np.array(training_label, dtype=np.int))
    return _knn


# KNN模型测试
def test_model_knn(_model, testing_data, testing_label):
    hit_cnt = 0
    # 使用训练后的模型预测分类
    for i in range(len(testing_label)):
        feature = np.array([testing_data[i]], dtype=np.float32)
        predict_id = _model.predict(feature)
        if predict_id == int(testing_label[i]):
            hit_cnt = hit_cnt + 1
    rate = hit_cnt / len(testing_label)
    return rate


# svm参数配置
def svm_config():
    _svm = cv2.ml.SVM_create()
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    _svm.setTermCriteria(criteria)
    _svm.setKernel(cv2.ml.SVM_LINEAR)
    _svm.setType(cv2.ml.SVM_C_SVC)
    _svm.setC(10000)
    return _svm


# SVM分类模型训练
def train_model_svm(training_data, training_label):
    # 创建svm分类器
    svm = svm_config()
    # 开始训练
    svm.train(np.array(training_data, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(training_label, dtype=np.int))
    return svm


# SVM模型测试
def test_model_svm(_model, testing_data, testing_label):
    hit_cnt = 0
    # 使用训练后的模型预测分类
    for i in range(len(testing_label)):
        feature = np.array([testing_data[i]], dtype=np.float32)
        _, predict_id = _model.predict(feature)
        if predict_id == int(testing_label[i]):
            hit_cnt = hit_cnt + 1
    rate = hit_cnt / len(testing_label)
    return rate


# 日志函数
def logger(msg):
    print(msg)


if __name__ == '__main__':
    # 读取类别信息
    CLASS_NUM, classes_info = load_dataset_info('./YALE_IMAGES')
    # 词典路径
    dictionary_path = './dictionary/local_descriptor/{0}_dictionary'.format(DESCRIPTOR_TYPE)
    if os.path.exists(dictionary_path):
        logger('Dictionary already exists, loading...')
        dictionary = np.loadtxt(dictionary_path)
    else:
        logger('Creating dictionary...')
        dictionary = build_dictionary()
        logger('Saving dictionary...')
        np.savetxt(dictionary_path, dictionary)
    print('Dictionary size: ', np.shape(dictionary))
    # 训练集、测试集路径
    dataset_path = './dataset/local_descriptor/{0}'.format(DESCRIPTOR_TYPE)
    if os.path.exists(dataset_path + '/train_data'):
        logger('Dataset already exists, loading...')
        # 从本地读取数据集
        train_data, train_label, test_data, test_label = read_from_local_dataset(dataset_path)
    else:
        logger('Building dataset...')
        os.makedirs(dataset_path, exist_ok=True)
        # 划分数据集
        train_data, train_label, test_data, test_label = build_dataset()
        logger('Saving dataset...')
        # 写入本地
        write_to_local_dataset(dataset_path, train_data, train_label, test_data, test_label)
    logger('Model training...')
    model = train_model_knn(train_data, train_label)
    # model = train_model_svm(train_data, train_label)
    logger('Model testing...')
    accuracy = test_model_knn(model, test_data, test_label)
    # accuracy = test_model_svm(model, test_data, test_label)
    print('The accuracy is %.2f%%' % (accuracy * 100))
