# -*- coding: utf-8 -*-
"""
 * @file            os_module.py
 * @brief           函数程序
 * @function        具体说明  
 * @compiler        Python 3.8 (64-bit) / PyCharm 
 * @author          Yan-kai Zhang (@Oliver_Zyk)
 * @version         V1.0 (Modified in 2023-05-07)
 * @date            2023-05-07
"""
"""
os常见的操作.

追求规范化、模块化、标准化.

Usage:
    from utils import os_module

    print(os_module.__doc__)
"""

import os
import shutil


def rm_if_exist_and_make(folder_list: list) -> None:
    '''删除并新建文件夹

    如果文件夹存在,   删除并重建;
    如果文件夹不存在, 新建;

    :param folder_list: 文件夹列表
    :return: None

    Usage:
        folder_list = [r'F:\tpz\paper\gen_bg_1', r'F:\tpz\paper\gen_bg_2']
        rm_if_exist_and_make(folder_list)
    '''

    for folder_path in folder_list:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  # delete folder
            os.makedirs(folder_path)  # make new folder
            print('%s 存在, 删除成功, 重新创建成功' % (folder_path))
        else:
            os.makedirs(folder_path)  # make new folder
            print('%s 不存在, 创建成功' % (folder_path))


def make_folder_if_no(folder_list: list) -> None:
    '''创建文件夹

    如果文件夹存在,   不做任何操作;
    如果文件夹不存在, 新建文件夹.

    :param folder_list: 文件夹列表
    :return: None

    Usage:
        folder_list = [r'F:\tpz\paper\gen_bg_1', r'F:\tpz\paper\gen_bg_2']
        make_folder_if_no(folder_list)
    '''

    for folder_path in folder_list:
        os.makedirs(folder_path, exist_ok=True)