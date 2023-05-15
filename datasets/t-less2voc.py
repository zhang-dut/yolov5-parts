# -*- coding: utf-8 -*-
"""
 * @file            t-less2voc.py
 * @brief           函数程序
 * @function        将 T-LESS 数据集转换成 yolov5 的数据集格式。
 * @compiler        Python 3.8 (64-bit) / PyCharm
 * @author          Yan-kai Zhang (@Oliver_Zyk)
 * @version         V1.0 (Modified in 2023-05-05)
 * @date            2023-05-05
"""
import os
import yaml
import shutil

import cv2

from utils.os_module import rm_if_exist_and_make
from utils.xml_module import generate_xml

Annotations_path = 'T-LESS_VOC/Annotations'
images_path = 'T-LESS_VOC/PNGImages'
rm_if_exist_and_make([Annotations_path, images_path])

t_less_path = 'T-LESS/'
folder_list = os.listdir(t_less_path)

folder_cnt = 0
for folder_name in folder_list:  # 分文件夹进行
    gt_file = os.path.join(t_less_path, folder_name, 'gt.yml')

    # 从gt中获取每幅图片的目标对象及bbox信息,k是图片名称,v是n个对象的bbox列表.
    gt_k_imgid_v_nbbox = {}

    with open(gt_file) as f:
        gt_dict = yaml.load(f, Loader=yaml.FullLoader)

    for k in list(gt_dict.keys())[:1295:5]:  # 第 k 幅图片, 前300张
        gt_k_imgid_v_nbbox[k] = []

        img_i_gt = gt_dict[k]
        # print(len(img_i_gt))  # 6

        img_name = '%s.png' % str(k).zfill(4)

        img_path = os.path.join(t_less_path, folder_name, 'rgb', img_name)
        print(img_path)
        img_size = cv2.imread(img_path).shape

        coords_list = []
        for j in range(len(img_i_gt)):
            # 获取第 i 幅图片第 j 个对象的 obj_bb
            img_i_gt_bb_j = img_i_gt[j]['obj_bb']  # obj_bb=[左上角x,左上角y, w,h]

            img_i_gt_bb_j = [  # [x1,y1,x2,y2] 是左上角和右下角的坐标
                img_i_gt_bb_j[0],  # x1
                img_i_gt_bb_j[1],  # y1
                img_i_gt_bb_j[0] + img_i_gt_bb_j[2],  # x2
                img_i_gt_bb_j[1] + img_i_gt_bb_j[3],  # y2
            ]

            img_i_gt_obj_id = img_i_gt[j]['obj_id']  # obj_id: 1

            coord = img_i_gt_bb_j + [str(img_i_gt_obj_id)]
            coords_list.append(coord)

        img_name_new = '%s.png' % str(k + folder_cnt * 1295).zfill(4)

        generate_xml(img_name_new, coords_list, img_size, Annotations_path)
        img_path_new = os.path.join(images_path, img_name_new)
        shutil.copyfile(img_path, img_path_new)  # 复制文件

    folder_cnt += 1