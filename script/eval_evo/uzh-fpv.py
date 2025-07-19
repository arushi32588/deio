import glob
import os
# from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import evo.main_ape as main_ape
import numpy as np
import torch

# 处理服务器中evo的可视化问题
import evo
from evo.tools.settings import SETTINGS
SETTINGS['plot_backend'] = 'Agg'

from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

# import sys
# sys.path.append('/home/gwp/DEIO2')#要导入

from devo.config import cfg # 这里的cfg是从devo.config中导入的
# from dpvo.dpvo import DPVO
# from dpvo.plot_utils import plot_trajectory
# from dpvo.stream import image_stream
from dpvo.utils import Timer

from utils.load_utils import load_gt_us,fpv_evs_iterator
from utils.eval_utils import log_results,compute_median_results,VO_run,EVO_run,EVO_run_GBA

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', default="datasets") # 数据集的路径
    parser.add_argument('--network', type=str, default='dpvo.pth') # 网络的路径
    parser.add_argument('--val_split', type=str, default="splits") # 验证集的路径,有它来决定验证的序列
    parser.add_argument('--config', default="config/***.yaml")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--enable_event', action="store_true")#是否启用事件,启用了后就不会再使用图像了
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--backend_thresh', type=float, default=64.0) # 用于判断是否使用后端优化的阈值，原本为64.0
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--side', type=str, default="left")

    parser.add_argument('--resnet', action='store_true', help='use the ResNet backbone')
    parser.add_argument('--block_dims', type=str, default="64,128,256", help='channel dimensions of ResNet blocks')
    parser.add_argument('--initial_dim', type=int, default=64, help='initial channel dimension of ResNet')
    parser.add_argument('--pretrain', type=str, default="resnet18", help='pretrained ResNet model (resnet18, resnet34)')

    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    # cfg.BACKEND_THRESH = args.backend_thresh # 用于判断是否使用后端优化的阈值，直接通过参数文件传入
    cfg.merge_from_list(args.opts)

    # 构造覆盖列表，将四个参数映射到cfg的键
    # 直接将args的参数赋值给cfg的顶层属性
    cfg.resnet = args.resnet
    cfg.block_dims = list(map(int, args.block_dims.split(',')))  # 转换为整数列表
    cfg.initial_dim = args.initial_dim
    cfg.pretrain = args.pretrain

    print("\033[42m Running EVO with config...\033[0m ")
    print(cfg, "\n")

    # torch.manual_seed(1234) #反而不利于同时测试多个数据集

    # 目前不要开启cfg.CLASSIC_LOOP_CLOSURE
    assert not cfg.CLASSIC_LOOP_CLOSURE #cfg.CLASSIC_LOOP_CLOSURE是用传统的方式来进行回环检测，现在不需要，只有相机临近法
    if cfg.LOOP_CLOSURE:
        print("\033[41m open the LOOP_CLOSURE with Global BA \033[0m ")
    else:
        print("\033[41m raw DEVO \033[0m ")

    # 读取场景的名称    
    test_scenes = open(args.val_split).read().split()
    print("the number of scenes is", len(test_scenes),"the input scenes are: ", test_scenes)

    dataset_name = "UZH-FPV"
    if args.enable_event:
        dataset_name += "/EVO"
    else:
        dataset_name += "/VO"
    
    if cfg.LOOP_CLOSURE:
        dataset_name += "_GBA"#如果开启了回环检测，就加上_GBA

    results_dict_scene, figures = {}, {}
    all_results = []
    for i, scene in enumerate(test_scenes):
        print(f"Eval on {scene}")
        results_dict_scene[scene] = []

        groundtruth = os.path.join(args.inputdir, scene, f"stamped_groundtruth_us.txt")#真值的路径

        for trial in range(args.trials):
            print(f"\nRunning trial {trial} of {scene}...")
            
            # 运行dpvo主程序
            if not args.enable_event:
                print("This code is for event rather than image")
            else:

                datapath_val = os.path.join(args.inputdir, scene)
                # load  traj（这应该是获取gt trajectory的值,从txt文件中读取）
                tss_traj_us, traj_hf = load_gt_us(groundtruth)#提前获取真值的时间戳和位置

                if cfg.LOOP_CLOSURE:
                    traj_est, tstamps, flowdata = EVO_run_GBA(datapath_val, cfg, args.network, viz=args.viz, 
                                            iterator=fpv_evs_iterator(datapath_val, stride=args.stride, timing=False, H=260, W=346,tss_gt_us=tss_traj_us),
                                            timing=False, H=260, W=346, viz_flow=False)
                else:
                    traj_est, tstamps, flowdata = EVO_run(datapath_val, cfg, args.network, viz=args.viz, 
                                            iterator=fpv_evs_iterator(datapath_val, stride=args.stride, timing=False, H=260, W=346,tss_gt_us=tss_traj_us),
                                            timing=False, H=260, W=346, viz_flow=False)

            # # load  traj（这应该是获取gt trajectory的值,从txt文件中读取）
            # tss_traj_us, traj_hf = load_gt_us(groundtruth)

            # do evaluation （进行验证）
            data = (traj_hf, tss_traj_us, traj_est, tstamps)
            hyperparam = (None, args.network, dataset_name, scene, trial, cfg, args)
            # 通过log_results函数来记录结果(用evo评估定位的精度)
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                   plot=True, save=True, return_figure=False, stride=args.stride,
                                                                   expname=scene,
                                                                   _n_to_align=-1
                                                                   )
            

        print(scene, sorted(results_dict_scene[scene]))
    
    results_dict = compute_median_results(results_dict_scene, all_results, dataset_name,outfolder=outfolder)

    for k in results_dict:
        print(k, results_dict[k])

    # print("AVG: ", np.mean(results_dict))

    

    
