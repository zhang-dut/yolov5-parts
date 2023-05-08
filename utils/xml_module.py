# -*- coding: utf-8 -*-
"""
 * @file            xml_module.py
 * @brief           函数程序
 * @function        具体说明  
 * @compiler        Python 3.8 (64-bit) / PyCharm 
 * @author          Yan-kai Zhang (@Oliver_Zyk)
 * @version         V1.0 (Modified in 2023-05-07)
 * @date            2023-05-07
"""
"""
目标检测中有关xml文件的常见操作.

追求规范化、模块化、标准化.

Usage:
    from utils import xml_module

    print(xml_module.__doc__)
"""

import os
import random

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom as DOC


def parse_xml(xml_path: str) -> list:
    ''' 从xml文件中提取bounding box信息

    :param xml_path: xml文件的路径
    :return: bboxes, 格式为[[x_min, y_min, x_max, y_max, name], ]

    Usage:
        xml_path='./data/test/Annotations\\0000.xml'
        bboxes = parse_xml(xml_path)
    '''

    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    bboxes = list()

    for ix, obj in enumerate(objs):
        # # 将不易识别和类别错误的去掉
        # difficult = obj.find('difficult').text
        # cls = obj.find('name').text
        # if cls not in classes or int(difficult) == 1:
        #     continue
        # # 将不易识别和类别错误的去掉

        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(box[0].text)
        y_min = int(box[1].text)
        x_max = int(box[2].text)
        y_max = int(box[3].text)
        bboxes.append([x_min, y_min, x_max, y_max, name])
    return bboxes


def generate_xml(img_name: str, bboxes: list, img_size: list, xml_save_path: str) -> None:
    '''将图片上所有用于目标检测的bounding box信息写入xml文件中.

    :param img_name: 图片名称
    :param bboxes: bounding box信息, [[x_min, y_min, x_max, y_max, name],], name为目标类别
    :param img_size:  [h,w,c], img_size=img.shape
    :param xml_save_path: xml文件保存路径
    :return: None

    Usage:
        img_name='0000.png'
        bboxes = [[511, 142, 587, 219, '1'], [273, 231, 336, 339, '4']]
        img_size = (540, 720, 3)
        xml_save_path = 'data/test/Annotations'
        generate_xml(img_name, bboxes, img_size, xml_save_path)
    '''

    doc = DOC.Document()  # 创建DOM文档对象

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    title = doc.createElement('folder')
    title_text = doc.createTextNode('Tianchi')
    title.appendChild(title_text)
    annotation.appendChild(title)

    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)

    source = doc.createElement('source')
    annotation.appendChild(source)

    title = doc.createElement('database')
    title_text = doc.createTextNode('The Tianchi Database')
    title.appendChild(title_text)
    source.appendChild(title)

    title = doc.createElement('annotation')
    title_text = doc.createTextNode('Tianchi')
    title.appendChild(title_text)
    source.appendChild(title)

    size = doc.createElement('size')
    annotation.appendChild(size)

    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)

    for bbox in bboxes:
        object = doc.createElement('object')
        annotation.appendChild(object)

        title = doc.createElement('name')
        title_text = doc.createTextNode(bbox[4])
        title.appendChild(title_text)
        object.appendChild(title)

        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        object.appendChild(difficult)

        bndbox = doc.createElement('bndbox')
        object.appendChild(bndbox)
        title = doc.createElement('xmin')
        title_text = doc.createTextNode(str(int(float(bbox[0]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymin')
        title_text = doc.createTextNode(str(int(float(bbox[1]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('xmax')
        title_text = doc.createTextNode(str(int(float(bbox[2]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymax')
        title_text = doc.createTextNode(str(int(float(bbox[3]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)

    # 将DOM对象doc写入文件
    f = open(os.path.join(xml_save_path, img_name[:-4] + '.xml'), 'w')
    f.write(doc.toprettyxml(indent=''))
    f.close()


def img_with_bboxes(img: np.ndarray, bboxes: list) -> np.ndarray:
    '''验证目标检测的标注信息是否正确

    :param img: img
    :param bboxes: bboxes信息, 格式为[[x_min, y_min, x_max, y_max],]
    :return: img

    Usage:
        img_path = 'data/test/images/0000.png'
        img = cv2.imread(img_path)
        xml_file = 'data/test/Annotations/0000.xml'
        bboxes = parse_xml(xml_file)  # 解析得到bboxes信息, 格式为[[x_min,y_min,x_max,y_max,name],]
        img_w_bboxes = img_with_bboxes(img, bboxes)
    '''

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        label = bbox[4]

        r = int(random.random() * 255)
        g = int(random.random() * 255)
        b = int(random.random() * 255)
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (b, g, r), 2)
        cv2.putText(img, '{}'.format(label), (x_min, y_min + 28), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (b, g, r), 2)

    return img