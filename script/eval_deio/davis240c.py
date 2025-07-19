import glob
import os
# from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import evo.main_ape as main_ape
import numpy as np
import torch
import quaternion
import math

# 处理服务器中evo的可视化问题
import evo
from evo.tools.settings import SETTINGS
SETTINGS['plot_backend'] = 'Agg'

from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from devo.config import cfg # 这里的cfg是从devo.config中导入的
# from dpvo.utils import Timer

from utils.load_utils import load_gt_us,davis240c_evs_iterator, davis240c_evs_h5_iterator
from utils.eval_utils import log_results,compute_median_results,VO_run,EVO_run,EVO_run_GBA,run_DEIO2

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
    parser.add_argument('--timing', action="store_true")

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
        print("\033[41m with Global BA \033[0m ")
    else:
        print("\033[41m no Global BA \033[0m ")

    # 读取场景的名称    
    test_scenes = open(args.val_split).read().split()
    print("the number of scenes is", len(test_scenes),"the input scenes are: ", test_scenes)

    dataset_name = "davis240c"
    if args.enable_event:
        dataset_name += "/EVO"
    else:
        dataset_name += "/VO"
    
    if cfg.LOOP_CLOSURE:
        dataset_name += "_GBA"#如果开启了回环检测，就加上_GBA
    
    if cfg.ENALBE_IMU:
        dataset_name += "_IMU"#如果还开启了IMU，就加上_IMU

    results_dict_scene, figures = {}, {}
    all_results = []
    for i, scene in enumerate(test_scenes):
        print(f"Eval on {scene}")
        results_dict_scene[scene] = []

        groundtruth = os.path.join(args.inputdir, scene, f"gt_stamped_left.txt")#真值的路径，注意这是有时间偏移的真值
        imupath = os.path.join(args.inputdir, scene, f"imu_data.csv")#IMU的路径

        for trial in range(args.trials):
            print(f"\nRunning trial {trial} of {scene}...")
            
            # 运行dpvo主程序
            if not args.enable_event:
                print("This code is for event rather than image")
                raise NotImplementedError

            datapath_val = os.path.join(args.inputdir, scene)
            # load  traj（这应该是获取gt trajectory的值,从txt文件中读取）
            tss_traj_us, traj_hf = load_gt_us(groundtruth)#提前获取真值的时间戳和位置

            if cfg.LOOP_CLOSURE and not cfg.ENALBE_IMU:#如果开启了回环检测，但是没有开启IMU
                # traj_est, tstamps, flowdata, avg_fps = EVO_run_GBA(datapath_val, cfg, args.network, viz=args.viz, 
                #                         iterator=davis240c_evs_iterator(datapath_val, side=args.side, stride=args.stride, timing=False, H=180, W=240),
                #                         timing=args.timing, H=180, W=240, viz_flow=False)
                raise NotImplementedError("No IMU, please check the config file")
            # elif cfg.LOOP_CLOSURE and cfg.ENALBE_IMU:#如果开启了回环检测，同时开启了IMU
            elif cfg.ENALBE_IMU:#如果开启了IMU （有无回环都可以运行这个）
                """ Load GT trajectory (for visualization and VI intilization) """
                # all_gt_keys=tss_traj_us #所有真值的时间戳
                # #all_gt为所有的真值的时间戳（tss_traj_us）+位姿（traj_hf）
                # all_gt = np.concatenate((all_gt_keys, traj_hf), axis=1)
                all_gt = {}#存真值的 时间戳+位姿
                # 遍历每个时间戳和对应的轨迹数据
                for sod, data in zip(tss_traj_us, traj_hf):
                    # sod是时间戳，为us，将us转换为秒
                    sod = float(sod / 1e6)
                    if sod not in all_gt:# 如果 sod 不在 all_gt 中，初始化一个空字典
                        all_gt[sod] = {}
                    
                    # 提取位置 (x, y, z)
                    x = data[0]
                    y = data[1]
                    z = data[2]
                    
                    # 提取四元数分量 (qx, qy, qz, qw)，注意GT就是这样存的
                    # 注意：根据实际数据中的四元数顺序调整索引
                    qx = data[3]
                    qy = data[4]
                    qz = data[5]
                    qw = data[6]
                    
                    # 构造四元数对象并转换为旋转矩阵
                    q = quaternion.from_float_array([float(qw), float(qx), float(qy), float(qz)])  # 注意四元数顺序是否为 (w, x, y, z)
                    R = quaternion.as_rotation_matrix(q)
                    
                    # 构造 4x4 变换矩阵
                    TTT = np.eye(4)
                    TTT[0:3, 0:3] = R
                    TTT[0:3, 3] = [float(x), float(y), float(z)]
                    
                    all_gt[sod]['T'] = TTT

                # 对时间戳进行排序*(注意这是有时间偏移的真值)
                all_gt_keys = sorted(all_gt.keys())#存存真值的时间戳，注意这里是秒
                assert np.all(all_gt_keys==tss_traj_us / 1e6)

                # t_offset_us = np.loadtxt(os.path.join(args.inputdir, scene, "t0_us.txt"))#读取时间偏移量
                raw_tss_imgs_ns = np.loadtxt(os.path.join(args.inputdir, scene, f"raw_tss_imgs_ns_left.txt"))#绝对时间戳
                raw_tss_imgs_us=raw_tss_imgs_ns/1e3#转换为微妙(us)
                tss_imgs_us = np.loadtxt(os.path.join(args.inputdir, scene, f"tss_imgs_us_left.txt"))#图像的时间戳(相对时间戳)
                t_offset_us=raw_tss_imgs_us[0]-tss_imgs_us[0]#第一帧的时间戳
                # assert t0_us == t_offset_us

                """ Load IMU data """
                all_imu = np.loadtxt(imupath,delimiter=',')#去掉第0列序号
                #将IMU的时间戳转换为微妙
                all_imu[:,0] /= 1e3 #读入的imu时间是纳秒，转换为微秒
                #将IMU的时间减去偏移量（这样可以跟图像的时间对齐，因为图像也有时间偏移量）
                all_imu[:,0] -= t_offset_us
                # 将时间小于0的数据去掉
                all_imu = all_imu[all_imu[:,0]>0]
                #将IMU的角速度转换为弧度
                all_imu[:,1:4] *= 180/math.pi
                # 将IMU根据时间戳进行排序，以此避免时间戳不连续的问题
                # 按照时间戳（第0列）排序
                all_imu = all_imu[all_imu[:, 0].argsort()]

                traj_est, tstamps, flowdata, avg_fps = run_DEIO2(datapath_val, cfg, args.network, viz=args.viz, 
                                        iterator=davis240c_evs_h5_iterator(datapath_val, side=args.side, stride=args.stride, timing=False, H=180, W=240),
                                        _all_imu=all_imu,
                                        _all_gt=all_gt,
                                        _all_gt_keys=all_gt_keys,
                                        timing=args.timing, H=180, W=240, viz_flow=False)

            else:
                # 报错
                # raise NotImplementedError("No loop closure and no IMU, please check the config file")
                traj_est, tstamps, flowdata, avg_fps = EVO_run(datapath_val, cfg, args.network, viz=args.viz, 
                                        iterator=davis240c_evs_h5_iterator(datapath_val, side=args.side, stride=args.stride, timing=False, H=180, W=240),
                                        timing=args.timing, H=180, W=240, viz_flow=False)

            # do evaluation （进行验证）
            data = (traj_hf, tss_traj_us, traj_est, tstamps)
            hyperparam = (None, args.network, dataset_name, scene, trial, cfg, args)
            # 通过log_results函数来记录结果(用evo评估定位的精度)
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                   plot=True, save=True, return_figure=False, stride=args.stride,
                                                                   expname=scene,
                                                                   _n_to_align=1000,
                                                                   avg_fps=avg_fps
                                                                   )
            
            gwp_debug=1;            

        print(scene, sorted(results_dict_scene[scene]))
    
    results_dict = compute_median_results(results_dict_scene, all_results, dataset_name,outfolder=outfolder)

    for k in results_dict:
        print(k, results_dict[k])

    print("Done!")

    

    
