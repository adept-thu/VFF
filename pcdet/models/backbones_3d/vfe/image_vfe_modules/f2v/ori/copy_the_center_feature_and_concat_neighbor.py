import imp
from typing_extensions import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
from mmcv.ops import ball_query
# from mmcv.ops import knn




from .point_to_image_projection import Point2ImageProjection

try:
    from kornia.losses.focal import BinaryFocalLossWithLogits
except:
    pass


class VoxelFieldFusion(nn.Module):
    def __init__(self, model_cfg, grid_size, pc_range, disc_cfg=None, device="cuda"):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            model_cfg: EasyDict, Module configuration
            grid_size: [X, Y, Z], Voxel grid size  单位为格
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        tensor_grid_size = torch.as_tensor(grid_size, dtype=torch.float32)
        self.pc_range = pc_range
        self.fuse_mode = model_cfg.get('FUSE', None)  # ray_sum
        self.fuse_stride = model_cfg.get('STRIDE', {})  # 下采样倍率1，2，4，8
        self.image_interp = model_cfg.get('INTERPOLATE', True)
        self.loss_cfg = model_cfg.get('LOSS', None)
        self.point_projector = Point2ImageProjection(grid_size=grid_size,
                                                     pc_range=pc_range,
                                                     fuse_mode=self.fuse_mode,
                                                     stride_dict=self.fuse_stride,
                                                     fuse_layer=model_cfg.LAYER_CHANNEL.keys())
        self.voxel_size = ((self.pc_range[1] - self.pc_range[0]) / tensor_grid_size)
        # self.pool = nn.AdaptiveAvgPool1d(3)
        if 'ray' in self.fuse_mode:
            # 初始化参数
            self.fuse_thres = model_cfg.get('FUSE_THRES', 0.5)
            self.depth_thres = model_cfg.get('DEPTH_THRES', 70)
            self.ray_thres = model_cfg.get('RAY_THRES', 1)
            self.block_num = model_cfg.get('BLOCK_NUM', 1)
            self.ray_sample = model_cfg.get('SAMPLE', {'METHOD': 'naive'})  # 'METHOD':learnable_uniform
            self.topk_ratio = model_cfg.get('TOPK_RATIO', 0.25)
            self.position_type = model_cfg.get('POSITION_TYPE', None)  # absolute

            # MLP
            self.ray_blocks = nn.ModuleDict()
            #TODO 
            self.pos_blocks = nn.ModuleDict()
            self.img_blocks = nn.ModuleDict()
            self.sample_blocks = nn.ModuleDict()
            self.fuse_blocks = nn.ModuleDict()
            # self.neighbor_fuse_blocks = nn.ModuleDict()

            self.judge_voxels = {}  # 判断voxel中是否有点

            self.kernel_size = self.loss_cfg.get("GT_KERNEL", 1)  # 3

            # 加载损失函数
            if 'FocalLoss' in self.loss_cfg.NAME:
                self.loss_func = BinaryFocalLossWithLogits(**self.loss_cfg.ARGS)
            else:
                raise NotImplementedError
            if 'BCELoss' in self.ray_sample.get("LOSS", "BCELoss"):
                self.sample_loss = torch.nn.BCEWithLogitsLoss()
            else:
                raise NotImplementedError

            for _layer in model_cfg.LAYER_CHANNEL.keys():  # 对每一层
                # 射线、图像、可学习采样都是OrderedDict形式
                ray_block = OrderedDict()
                #TODO pos_block
                pos_block = OrderedDict()
                img_block = OrderedDict()
                sample_blocks = OrderedDict()

                ray_in_channel, img_in_channel = 3, model_cfg.LAYER_CHANNEL[_layer]  # 只有"layer1": 16
                out_channel = model_cfg.LAYER_CHANNEL[_layer]
                #  sparse_shape:grid的大小
                sparse_shape = np.ceil(self.grid_size / self.fuse_stride[_layer]).astype(int)  # ceil向上取整
                sparse_shape = sparse_shape[::-1] + [1, 0, 0]  # [::-1]倒序 ZYX ->XYZ

                # self.judge_voxels生成对应xyz大小的空tenser
                self.judge_voxels[_layer] = torch.zeros(*sparse_shape).to(device=device)
                if self.position_type is not None:
                    img_in_channel += 2  # 3+2=5
                for _block in range(self.block_num):  # 每一层的各个块（3块）
                    ray_block['ray_{}_conv_{}'.format(_layer, _block)] = nn.Linear(in_features=ray_in_channel,
                                                                                   out_features=out_channel,
                                                                                   bias=True)
                    pos_block['ray_{}_conv_{}'.format(_layer, _block)] = nn.Linear(in_features=3,
                                                                                   out_features=out_channel,
                                                                                   bias=True)
                    img_block['img_{}_conv_{}'.format(_layer, _block)] = nn.Conv2d(in_channels=img_in_channel,
                                                                                   out_channels=out_channel,
                                                                                   kernel_size=1,
                                                                                   stride=1,
                                                                                   padding=0,
                                                                                   bias=True)
                    if "learnable" in self.ray_sample.METHOD:  # importance采样法
                        sample_blocks['sample_{}_conv_{}'.format(_layer, _block)] = nn.Conv2d(
                            in_channels=img_in_channel,
                            out_channels=out_channel if _block < self.block_num - 1 else 1,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True)
                    if _block < self.block_num - 1:
                        ray_block['ray_{}_relu_{}'.format(_layer, _block)] = nn.ReLU()
                        pos_block['ray_{}_relu_{}'.format(_layer, _block)] = nn.ReLU()
                        img_block['img_{}_bn_{}'.format(_layer, _block)] = nn.BatchNorm2d(out_channel)
                        img_block['img_{}_relu_{}'.format(_layer, _block)] = nn.ReLU()
                        if "learnable" in self.ray_sample.METHOD:
                            sample_blocks['sample_{}_bn_{}'.format(_layer, _block)] = nn.BatchNorm2d(out_channel)
                            sample_blocks['sample_{}_relu_{}'.format(_layer, _block)] = nn.ReLU()
                    ray_in_channel = out_channel
                    img_in_channel = out_channel

                # weight init
                for _ray in ray_block:
                    if 'relu' in _ray or 'bn' in _ray: continue
                    nn.init.normal_(ray_block[_ray].weight, mean=0, std=0.01)
                    if ray_block[_ray].bias is not None:
                        nn.init.constant_(ray_block[_ray].bias, 0)
                for _pos in pos_block:
                    if 'relu' in _pos or 'bn' in _pos: continue
                    nn.init.normal_(pos_block[_pos].weight, mean=0, std=0.01)
                    if pos_block[_pos].bias is not None:
                        nn.init.constant_(pos_block[_ray].bias, 0)                
                for _img in img_block:
                    if 'relu' in _img or 'bn' in _img: continue
                    nn.init.normal_(img_block[_img].weight, mean=0, std=0.01)
                    if img_block[_img].bias is not None:
                        nn.init.constant_(img_block[_img].bias, 0)
                if "learnable" in self.ray_sample.METHOD:
                    for _sample in sample_blocks:
                        if 'relu' in _sample or 'bn' in _sample: continue
                        nn.init.normal_(sample_blocks[_sample].weight, mean=0, std=0.01)
                        if sample_blocks[_sample].bias is not None:
                            nn.init.constant_(sample_blocks[_sample].bias, 0)
                    self.sample_blocks[_layer] = nn.Sequential(sample_blocks)

                #  三个embeding层，都是MLP
                self.ray_blocks[_layer] = nn.Sequential(ray_block)
                self.pos_blocks[_layer] = nn.Sequential(pos_block)
                self.img_blocks[_layer] = nn.Sequential(img_block)
                self.fuse_blocks[_layer] = nn.Sequential(nn.Linear(in_features=out_channel * 3,
                                                                   out_features=out_channel,
                                                                   bias=True),
                                                         nn.ReLU())
                # self.neighbor_fuse_blocks[_layer] = nn.Sequential(nn.Linear(in_features=out_channel * 2,
                #                                                    out_features=out_channel,
                #                                                    bias=True),
                #                                          nn.ReLU())
                

    def position_encoding(self, H, W):
        if self.position_type == "absolute":
            min_value = (0, 0)
            max_value = (W - 1, H - 1)
        elif self.position_type == "relative":
            min_value = (-1.0, -1.0)
            max_value = (1.0, 1.0)

        # torch.linspace返回一个1维张量，包含在区间start和end上均匀间隔的step个点。
        loc_w = torch.linspace(min_value[0], max_value[0], W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(min_value[1], max_value[1], H).cuda().unsqueeze(1).repeat(1, W)
        # 两张表大小都是H*W上下拼接2*H*W
        loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
        return loc

    def square_distance(self, src, dst):
        """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src.float(), dst.float().permute(0, 2, 1))
        dist += torch.sum(src.float()** 2, -1).view(B, N, 1)
        dist += torch.sum(dst.float()** 2, -1).view(B, 1, M)
        return dist

    def index_points(self, points, idx):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """

        # device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        # batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx.long(), :]
        return new_points

    def fill(self, k, encoded_voxel_batch, select_voxel_batch):
        dist, neighbor_idx = self.knn_point(k, encoded_voxel_batch, select_voxel_batch)  # B,N_idx,XYZ
        
        grouped_encoded_voxel_for_each_render_point = self.index_points(encoded_voxel_batch,
                                                                        neighbor_idx)  # B,N,k
        
        max_pos = torch.max(grouped_encoded_voxel_for_each_render_point, dim=2)[0]  # BN3
        min_pos = torch.min(grouped_encoded_voxel_for_each_render_point, dim=2)[0]  # BN3

        return max_pos, min_pos


    def fusion(self, image_feat, voxel_feat, image_grid):  # 对应公式(4)
        """
        Fuses voxel features and image features 在通道维度上相加
        Args:
            image_feat: (C, H, W), Encoded image features
            voxel_feat: (N, C), Encoded voxel features
            image_grid: (N, 2), Image coordinates in X,Y of image plane
        Returns:
            voxel_feat: (N, C), Fused voxel features
        """
        image_grid = image_grid[:, [1, 0]]  # X,Y -> Y,X

        if 'sum' in self.fuse_mode:
            fuse_feat = image_feat[:, image_grid[:, 0], image_grid[:, 1]]
            voxel_feat += fuse_feat.permute(1, 0)
        elif 'mean' in self.fuse_mode:
            fuse_feat = image_feat[:, image_grid[:, 0], image_grid[:, 1]]
            voxel_feat = (voxel_feat + fuse_feat.permute(1, 0)) / 2
        else:
            raise NotImplementedError

        return voxel_feat

    def forward(self, batch_dict, encoded_voxel=None, encoded_feat2d=None, layer_name=None):
        """
        Generates voxel features via 3D transformation and sampling
        encoded_voxel：已经得到的点
        batch_dict  含有转化矩阵
            grid_to_lidar: (4, 4), Voxel grid to LiDAR unprojection matrix
            lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
            cam_to_img: (B, 3, 4), Camera projection matrix

        注意，输入的点的encoded_voxel.indice是(N,BZYX)形式，所有batch的点都排列在一起，每个点有自己所在的batch的idx
        ，储存在B这个维度上
        """
        # Generate sampling grid for frustum volume  点和grid分别投影
        projection_dict = self.point_projector(voxel_coords=encoded_voxel.indices.float(),
                                               batch_dict=batch_dict,
                                               layer_name=layer_name)

        ray_pred, ray_gt, ray_multi, sample_pred, sample_gt = [], [], [], [], []

        for _idx in range(len(batch_dict['image_shape'])):  # _idx之的是batch的idx,对每个batch有
            if encoded_feat2d is None:
                image_feat = batch_dict['image_features'][_idx]
            else:
                image_feat = encoded_feat2d[_idx]
            # TODO                
            # image_feat_npy = image_feat[0,:,:].detach().cpu().numpy()
            # import matplotlib
            # matplotlib.image.imsave('/home/zhanghaoming/visual/input_image.png', image_feat_npy)
            # raw_shape指每一层下采样后的大小
            raw_shape = tuple(batch_dict['image_shape'][_idx].cpu().numpy() // self.fuse_stride[layer_name])
            feat_shape = image_feat.shape[-2:]
            if self.image_interp:  # 如果要插值
                image_feat = F.interpolate(image_feat.unsqueeze(0), size=raw_shape, mode='bilinear')[0]
            # 把这一batch的点找出来
            index_mask = encoded_voxel.indices[:, 0] == _idx  # encoded_voxel.indices: N, BZYX
            # 这里这一batch点的信息和投影到图片上的的信息找出来
            voxel_feat = encoded_voxel.features[index_mask]  # 这一batch的点的features
            image_grid = projection_dict['image_grid'][_idx]  # encoded_voxel投影在格子上的xy坐标 (B,N,2)
            point_mask = projection_dict['point_mask'][_idx]  # 每个batch的点被置1 
            image_depth = projection_dict['image_depths'][_idx]  # 点投影在格子上的深度
            point_inv = projection_dict['point_inv'][_idx]  # 真实点的lidar坐标

            # Fuse 3D LiDAR point with 2D image feature
            # point_mask[len(voxel_feat):] -> 0 for batch construction
            # 找出这一batch中的point_feat
            voxel_mask = point_mask[:len(voxel_feat)]  # 找出这一batch中的point，做成mask
            if self.training and 'overlap_mask' in batch_dict.keys():
                overlap_mask = batch_dict['overlap_mask'][_idx]
                is_overlap = overlap_mask[image_grid[:, 1], image_grid[:, 0]].bool()  # 如果投影到图片上的坐标x,y重叠了
                if 'depth_mask' in batch_dict.keys():
                    depth_mask = batch_dict['depth_mask'][_idx]
                    depth_range = depth_mask[image_grid[:, 1], image_grid[:, 0]]
                    is_inrange = (image_depth > depth_range[:, 0]) & (image_depth < depth_range[:, 1])
                    is_overlap = is_overlap & (~is_inrange)

                image_grid = image_grid[~is_overlap]
                point_mask = point_mask[~is_overlap]
                voxel_mask = voxel_mask & (~is_overlap[:len(voxel_feat)])
            if not self.image_interp:
                image_grid = image_grid.float()
                image_grid[:, 0] *= (feat_shape[1] / raw_shape[1])
                image_grid[:, 1] *= (feat_shape[0] / raw_shape[0])
                image_grid = image_grid.long()
            voxel_feat[voxel_mask] = self.fusion(image_feat, voxel_feat[voxel_mask],
                                                 image_grid[point_mask])  # 图片信息与点的信息融合
            # voxel_feat[voxel_mask]和image_grid[point_mask]时一一对应的关系
            
            encoded_voxel.features[index_mask] = voxel_feat
            # TODO在这里加入了坐标与feat融合
            # 这一batch的信息融合结束 encoded_voxel.features(N,C)

            # Predict 3D Ray from 2D image feature
            if 'ray' in self.fuse_mode:
                # Get projected variables for ray rendering
                ray_mask = projection_dict['ray_mask'][_idx]  # 用于切割grid生成的深度图像的mask
                ray_depth = projection_dict['ray_depths'][_idx]  # grid生成的深度图像的深度
                ray_mask = ray_mask & (ray_depth < self.depth_thres)
                ray_voxel = projection_dict['voxel_grid'][_idx][ray_mask]  # mask切割后的gird三维坐标 N, ZYX
                ray_grid = projection_dict['ray_grid'][_idx][ray_mask]  # mask切割后的grid生成的深度图,ray_grid 是voxel块（体素场）在图片上的投影
                lidar_grid = projection_dict['lidar_grid'][_idx][ray_mask]  # mask切割后的grid生成的lidar坐标系下的坐标

                # Get shape of render voxel and grid
                render_shape = batch_dict['image_shape'][_idx] // self.fuse_stride[layer_name]
                render_shape = render_shape.flip(dims=[0]).unsqueeze(0)

                # Add positional embedding if needed
                if self.position_type is not None:
                    H, W = image_feat.shape[-2:]
                    if self.position_type is not None:
                        pos_embedding = self.position_encoding(H=H, W=W)
                    image_feat = torch.cat([image_feat, pos_embedding.squeeze()], dim=0)

                # Paint GT Voxel GT 也就是所有包含点的Voxel置1
                # encoded_voxel.indices(N,BXYZ))
                voxel_indices = encoded_voxel.indices[index_mask][:, 1:].long()  # 这一步骤筛选出batch包含的点的XYZ坐标
                # 空gird, 每个格子的信息不是坐标，而是0,1，作为mask
                judge_voxel = self.judge_voxels[layer_name] * 0
                # 有实际点的格子置1
                judge_voxel[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1

                # Select TOP number
                topk_num = int(encoded_voxel.indices.shape[0] * self.topk_ratio)
                # topk_num = 20
                """
                #利用TOP number生成射线 
                输入：
                ray_grid 是voxel块（体素场）在图片上的投影,
                lidar_grid是grid在lidar坐标系下的坐标                   
                
                输出：
                render_feat：图片和点融合后的特征
                ray_logit：不加sigmoid的grid_prob
                sample_mask：可学习的采样法中，获得较高得分的，体素投影到图片的点的mask
                ray_mask：ray_prob(omega_j)得分较高的投影点的mask
                grid_prob：可学习采样法得到的图片上的像素的分数           
                """
                render_feat, ray_logit, sample_mask, ray_mask, grid_prob, sample_encoded_mask = self.ray_render(ray_grid, lidar_grid,
                                                                                           image_grid[point_mask],
                                                                                           image_feat.unsqueeze(0),
                                                                                           render_shape, layer_name,
                                                                                           topk_num,
                                                                                           voxel_indices = voxel_indices,
                                                                                           encoded_voxel_feat = voxel_feat[voxel_mask])

                # Find the pair of rendered voxel and orignial voxel
                render_indices = ray_voxel[sample_mask][ray_mask][:, [2, 1, 0]].long()  # 被选出的voxel的grid坐标 N,XYZ
                # np.savetxt('/home/zhanghaoming/visual/np_render_indices.txt', render_indices.cpu().numpy())
                # 被选出的voxel中是否包含encoded_voxel，若包含，置1，生成render_mask
                # TODO
                voxel_indices_select= voxel_indices[voxel_mask][sample_encoded_mask]
                voxel_feat_select = voxel_feat[voxel_mask][sample_encoded_mask]

##########################TODO
                # 这一batch中所有点的坐标(N,XYZ)         
                k = 8 //self.fuse_stride[layer_name]
                r = 10 //self.fuse_stride[layer_name]
                if render_indices.size(0) !=0 and voxel_indices_select.size(0) !=0:
                    ball_index = ball_query(0,r,k,render_indices.unsqueeze(0).float(),voxel_indices_select.unsqueeze(0).float())
                    render_indices  = self.index_points(render_indices.unsqueeze(0),ball_index).squeeze(0).reshape(-1,3)
                    # render_feat = self.index_points(render_feat.unsqueeze(0),ball_index).squeeze(0).reshape(-1,render_feat.size(-1)) # 真实xyz的坐标对应的特征
                    render_feat = self.index_points(render_feat.unsqueeze(0),ball_index).squeeze(0) # 真实xyz的坐标对应的特征
                    render_feat = torch.cat([voxel_feat_select.unsqueeze(1).repeat(1,k,1),render_feat],dim = -1).reshape(-1,3 * voxel_feat_select.size(-1))
                    # render_feat = (voxel_feat_select.repeat(1,k,1) *   
                    #                 (self.index_points(render_feat.unsqueeze(0),ball_index).squeeze(0))).reshape(-1,render_feat.size(-1))
                    render_feat = self.fuse_blocks[layer_name](render_feat)
##########################

                render_mask = judge_voxel[render_indices[:, 0], render_indices[:, 1], render_indices[:, 2]].bool()
                judge_voxel = judge_voxel * 0

                judge_voxel[
                    render_indices[render_mask][:, 0], render_indices[render_mask][:, 1], render_indices[render_mask][:,
                                                                                          2]] = 1
                # 同时符合以下条件的voxel的置1
                # 条件1：voxel被ray_render筛选出来
                # 条件2：筛选出来的voxel中含被选出的voxel中是否包含encoded_voxel
                voxel_mask = judge_voxel[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]].bool()

                # Add rendered point to Sparse Tensor
                render_indices = render_indices[~render_mask].int()  # 选出不包含encoded_voxel点的voxel的坐标
                render_feat = render_feat[~render_mask]  # 选出不包含encoded_voxel的voxel的特征
                # vis
                # np.savetxt('/home/zhanghaoming/visual/np_ball_render.txt',render_indices.reshape(-1,3).cpu().numpy())
                # np.savetxt('/home/zhanghaoming/visual/np_encoded_voxel.txt', voxel_indices.cpu().numpy())
                # np.savetxt('/home/zhanghaoming/visual/np_foreground_voxel.txt',voxel_indices_select.cpu().numpy())

                if layer_name == 'layer1' or layer_name == 'layer2': # TODO
                # _idx之的是batch的idx 
                    if render_indices.size(0) !=0 and voxel_indices_select.size(0) !=0:  
                        render_batch = _idx * torch.ones(len(render_indices), 1).to(device=render_indices.device).int()
                        # 给render_indices(xyz)加上batch编号
                        render_indices = torch.cat([render_batch, render_indices], dim=1)
                        # 把ray_render筛选出来的新的带有特征的voxel加入到encoded_voxel里
                        encoded_voxel.indices = torch.cat([encoded_voxel.indices, render_indices], dim=0)
                        # encoded_voxel.features = torch.cat([encoded_voxel.features, render_feat], dim=0)
                        encoded_voxel=encoded_voxel.replace_feature(torch.cat([encoded_voxel.features, render_feat], dim=0))
                if not self.training:
                    continue

                # Find the points in GT ray(这一段就是找出投影在前景区域的真实点)
                grid_mask = torch.zeros(tuple(render_shape[0].cpu().numpy())).to(device=image_grid.device)
                grid_mask[image_grid[point_mask][:, 0], image_grid[point_mask][:, 1]] = 1  # 点投影到图片上的像素置1
                # （这句话写的不好，看上面的注释）即是点投影到图片上的像素，又是ray_render筛选出来的voxel投影得到的像素，设置为identity_mask
                identity_mask = grid_mask[ray_grid[sample_mask][:, 0], ray_grid[sample_mask][:, 1]].bool()

                # Find the pair of rendered voxel and orignial voxel
                # 以一个encoded_voxel为中心形成3D高斯场（是否可以改成只在前景点上生成高斯场）
                judge_voxel = judge_voxel * 0
                if self.kernel_size > 1:
                    judge_voxel = self.gaussian3D(judge_voxel, voxel_indices)
                else:
                    judge_voxel[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1

                # 这一步要选出满足如下条件的点（作为真值）
                # 该点的投影属于前景点，且在射线上
                render_indices = ray_voxel[sample_mask][identity_mask][:, [2, 1, 0]].long()
                # np.savetxt('/home/zhanghaoming/visual/identity_mask.txt', render_indices.cpu().numpy())
                render_mask = judge_voxel[render_indices[:, 0], render_indices[:, 1], render_indices[:, 2]].float()

                if identity_mask.sum() > 0:
                    ray_logit = ray_logit[identity_mask]  # 用identity_mask筛选出voxel的分数
                else:
                    # Avoid Loss Nan
                    ray_logit = ray_logit.sum()[None] * 0
                    render_mask = render_mask.sum()[None] * 0
                ray_pred.append(ray_logit)
                ray_gt.append(render_mask)

                # 对于图片中的像素的得分 grid_prob形状(1BHW)
                if grid_prob is not None:
                    grid_prob = grid_prob[0]
                    grid_gt = torch.zeros_like(grid_prob)
                    box2d_gt = batch_dict['gt_boxes2d'][_idx]

                    if "affine_matrix" in batch_dict:  # 如果含有仿射矩阵，则变换
                        box2d_gt = box2d_gt.reshape(-1, 2, 2)
                        box2d_gt = torch.cat([box2d_gt, torch.ones(box2d_gt.shape[0], 2, 1).to(box2d_gt.device)],
                                             dim=-1)
                        box2d_T = box2d_gt @ batch_dict["affine_matrix"][_idx, :2].T
                        norm_T = box2d_gt @ batch_dict["affine_matrix"][_idx, -1].T
                        box2d_gt = (box2d_T / norm_T[..., None])
                        box2d_gt[..., 0] = box2d_gt[..., 0].clip(min=0, max=raw_shape[1])
                        box2d_gt[..., 1] = box2d_gt[..., 1].clip(min=0, max=raw_shape[0])
                        box2d_gt = box2d_gt.reshape(-1, 4)

                    box2d_gt = (box2d_gt // self.fuse_stride[layer_name]).long()
                    if "gaussian" in self.ray_sample.get("GT_TYPE", "box"):
                        grid_gt = self.gaussian2D(grid_gt, box2d_gt)
                    else:
                        for _box in box2d_gt:
                            grid_gt[:, _box[1]:_box[3], _box[0]:_box[2]] = 1
                    sample_pred.append(grid_prob)
                    sample_gt.append(grid_gt)

        if self.training and self.loss_cfg is not None:
            ray_pred = torch.cat(ray_pred, dim=0)
            ray_gt = torch.cat(ray_gt, dim=0)
            if len(ray_multi) > 0:
                ray_multi = torch.cat(ray_multi, dim=0)
            if len(sample_pred) > 0:
                sample_pred = torch.cat(sample_pred, dim=0)
                sample_gt = torch.cat(sample_gt, dim=0)
            loss_dict = self.get_loss(ray_pred, ray_gt, ray_multi, sample_pred, sample_gt)
            for _key in loss_dict:
                batch_dict[_key + '_' + layer_name] = loss_dict[_key]

        return encoded_voxel, batch_dict

    def ray_render(self, ray_grid, ray_feat, image_grid, image_feat, shape, layer_name, topk_num,voxel_indices =None, encoded_voxel_feat = None, min_n=-1, max_n=1):
        """
        Args:
            ray_grid:  lidar_grid投影在图片上的坐标
            ray_feat: lidar 坐标
            image_grid: 真实点投影在图片上的xy坐标
            image_feat:  图片本身的特征(C, H, W), Encoded image features
            shape: 图片大小
            layer_name:
            topk_num:
            min_n:
            max_n:

        Returns:

        """
        # 把图片划分成若干个windows（如64*64），在每个windows里取一定数量的点
        grid_prob = None
        window_size = self.ray_sample.WINDOW // self.fuse_stride[layer_name]  # self.ray_sample.WINDOW=64
        # .ceil()向上取整
        grid_x = torch.arange(0, ((shape[0, 0] / window_size).ceil() + 1) * window_size + 1, step=window_size)  # 网格x轴被划分为 0,64,128...2048
        range_x = torch.stack([grid_x[:-1], grid_x[1:] - 1]).transpose(0, 1).to(device=image_grid.device)  # 网格x轴每个格子范围 [0,63],[64,127]...
        grid_y = torch.arange(0, ((shape[0, 1] / window_size).ceil() + 1) * window_size + 1, step=window_size)  # 同理
        range_y = torch.stack([grid_y[:-1], grid_y[1:] - 1]).transpose(0, 1).to(device=image_grid.device)  # 同理
        # 根据点投影在图片上的坐标作出mask
        mask_x = (image_grid[:, 0][None, :] >= range_x[:, 0][:, None]) & \
                 (image_grid[:, 0][None, :] <= range_x[:, 1][:, None])  # 每个网格筛选出包含于其中的image_grid
        mask_y = (image_grid[:, 1][None, :] >= range_y[:, 0][:, None]) & \
                 (image_grid[:, 1][None, :] <= range_y[:, 1][:, None])  # 网格筛选出包含于其中的image_grid
        grid_mask = mask_x[:, None, :] & mask_y[None, :, :]
        grid_count = grid_mask.sum(-1)  # 每一个网格中有几个真实点

        if "uniform" in self.ray_sample.METHOD:
            sample_num = len(image_grid) * self.ray_sample.RATIO
            grid_count = (grid_count > 0) * sample_num // (grid_count > 0).sum()
            # 在所有有点的格子中均匀地采样出sample_num个点，每个格子中的点为grid_count个
        elif "density" in self.ray_sample.METHOD:
            grid_count = grid_count * self.ray_sample.RATIO
            # 在所有有点的格子中按照密度采样出sample_num个点，每个格子中的点数与格子中的真实点数成正比
        elif "sparsity" in self.ray_sample.METHOD:
            sample_count = grid_count[grid_count > 0]
            sample_num, sample_idx = (sample_count * self.ray_sample.RATIO).sort() #从小到大排序
            sample_count[sample_idx] = sample_num.long().flip(0) # 反转顺序
            grid_count[grid_count > 0] = sample_count
            # 在所有有点的格子中按照稀疏程度采样出sample_num个点，每个格子中的点数与格子中的真实点数成正比
        if "all" in self.ray_sample.METHOD:
            grid_sample = torch.ones(grid_count.shape[0] * window_size, grid_count.shape[1] * window_size).bool().to(
                device=image_grid.device)
        else:
            grid_sample = torch.rand(*grid_count.shape, window_size, window_size).to(device=image_grid.device)
            grid_ratio = grid_count / (window_size ** 2)  # 每格中点的数量/格子大小 [Nx,Ny]
            grid_sample = grid_sample < grid_ratio[..., None, None]
            grid_sample = grid_sample.permute(0, 2, 1, 3)
            grid_sample = grid_sample.reshape(grid_sample.shape[0] * window_size, -1)
            # vis
            # matplotlib.image.imsave('/home/zhanghaoming/visual/grid_sample0.png', grid_sample.int().float().detach().cpu().numpy())


        if "learnable" in self.ray_sample.METHOD:  # 可学习的采样法
            grid_prob = self.sample_blocks[layer_name](image_feat)  # sample_blocks是一个2d卷积
            # grid_mask 指图片特征经过MLP后grid_prob大于阈值的点,阈值为0.5
            grid_mask = (grid_prob.sigmoid() > self.ray_sample.THRES).squeeze() # 对应公式(2)
            grid_mask = grid_mask.transpose(0, 1)
            # vis
            # matplotlib.image.imsave('/home/zhanghaoming/visual/name_mask.png', grid_mask.int().float().detach().cpu().numpy())
            
            # 用获得高分的grid_prob制成的的grid_mask裁减出用于生成ray的点生成grid_sample，对应公式(2)
            grid_sample[:grid_mask.shape[0], :grid_mask.shape[1]] = grid_mask & grid_sample[:grid_mask.shape[0],
                                                                                :grid_mask.shape[1]]
            # TODO  
            grid_sample_true = torch.zeros_like(grid_sample) 
            grid_sample_true[:grid_mask.shape[0], :grid_mask.shape[1]] = 1
            grid_sample_true[:grid_mask.shape[0], :grid_mask.shape[1]] = grid_mask & grid_sample_true[:grid_mask.shape[0], :grid_mask.shape[1]]                                              
        # vis
        # matplotlib.image.imsave('/home/zhanghaoming/visual/grid_sample.png', grid_sample_true.int().float().detach().cpu().numpy())
        # TODO修改采样法
        sample_encoded_mask = grid_sample_true[image_grid[:, 0], image_grid[:, 1]]
        image_grid_select = image_grid[sample_encoded_mask]
        encoded_voxel_feat_select = encoded_voxel_feat[sample_encoded_mask]
        
        # sample_mask指的是grid_sample中包含grid投影的点
        sample_mask = grid_sample[ray_grid[:, 0], ray_grid[:, 1]]
        # 从ray_grid选出用于生成ray的点
        # 这些点满足以下条件：1.在图片处理中获得高分的像素 2.有点投影在这些获得高分的像素上
        ray_grid = ray_grid[sample_mask]
        ray_feat = ray_feat[sample_mask]

        # Get feature embedding  利用mlp处理grid采样图片中的关键点
        image_feat = self.img_blocks[layer_name](image_feat)
        ray_feat = self.ray_blocks[layer_name](ray_feat)

        # Subtract 1 since pixel indexing from [0, shape - 1]
        norm_coords = ray_grid / (shape - 1) * (max_n - min_n) + min_n
        norm_coords = norm_coords.reshape(1, 1, -1, 2)
        # grid_feat此处定义：指的是图片正规化采样后的特征 注意是BNC形状
        grid_feat = F.grid_sample(input=image_feat, grid=norm_coords, mode="bilinear", padding_mode="zeros")
        grid_feat = grid_feat[0, :, 0].transpose(0, 1)  # 得到每个投影点的新特征(N,C)
        ray_logit = (ray_feat * grid_feat).sum(-1)  # ray_feat * grid_feat并按通道相加，
        ray_prob = ray_logit.sigmoid()  # 这里对应公式(3)

        if self.training:
            if len(ray_prob) > topk_num:
                ray_topk = torch.topk(ray_prob, topk_num)[1]  # 找到得分最高的前k个ray_prob的idx
                # 用前k个ray_prob的值的idx做出ray_mask
                ray_mask = torch.zeros_like(ray_prob).bool()
                ray_mask[ray_topk] = True
            else:
                ray_mask = torch.ones_like(ray_prob).bool()
        else:  # test模式
            ray_mask = (ray_prob > self.fuse_thres)  # 得分高于阈值的点制作成mask
            if ray_mask.sum() > topk_num:  # 如果得分高于阈值的点多于设定值，进一步筛选
                ray_topk = torch.topk(ray_prob, topk_num)[1]
                top_mask = torch.zeros_like(ray_prob).bool()
                top_mask[ray_topk] = True
                ray_mask = ray_mask & top_mask

        # 用ray_mask筛选出得分高的ray_prob的值
        ray_prob = ray_prob[ray_mask]

        
        # 拼接lidar特征和图片特征
        render_feat = torch.cat([ray_feat[ray_mask], grid_feat[ray_mask]], dim=1) #2C
        # render_feat = self.fuse_blocks[layer_name](render_feat)
        # MLP把通道数压缩为原来的一半（保证输出的通道与原始通道数相同）
        render_feat = render_feat * ray_prob.unsqueeze(-1)

        return render_feat, ray_logit, sample_mask, ray_mask, grid_prob, sample_encoded_mask


    """
    render_feat：图片和点融合后的特征
    ray_logit：不加sigmoid的ray_prob
    sample_mask：可学习的采样法中，获得较高得分的，体素投影到图片的点的mask
    ray_mask：ray_prob(omega_j)得分较高的投影点的mask
    ray_prob：图片特征中每个像素-点响应,对应公式三中的omega_j
    """

    def get_loss(self, ray_pred, ray_gt, ray_multi, sample_pred, sample_gt):
        loss_dict = {}
        loss_ray = self.loss_func(ray_pred[None, None, :], ray_gt[None, None, :])
        if self.loss_cfg.ARGS["reduction"] == "sum":
            loss_ray = loss_ray / max((ray_gt == 1).sum(), 1)
        if len(ray_multi) > 0:
            loss_dict["ray_loss"] = self.loss_cfg.WEIGHT * (ray_multi * loss_ray.squeeze()).mean()
        else:
            loss_dict["ray_loss"] = self.loss_cfg.WEIGHT * loss_ray
        if len(sample_pred) > 0:
            loss_sample = self.sample_loss(sample_pred, sample_gt)
            loss_dict["sample_loss"] = self.ray_sample.WEIGHT * loss_sample
        return loss_dict

    def gaussian3D(self, voxel, grid, sigma_factor=3):
        """
        对应公式(6)
        Args:
            voxel:
            grid:
            sigma_factor:

        Returns:

        """
        kernel = self.kernel_size // 2
        sigma = self.kernel_size / sigma_factor
        loc_range = np.linspace(-kernel, kernel, self.kernel_size).astype(np.int)  # [-1,0,1]
        max_shape = voxel.shape
        for _x in loc_range:
            for _y in loc_range:
                for _z in loc_range:
                    # generate gaussian-like GT with a factor sigma
                    gauss = np.exp(-(_x * _x + _y * _y + _z * _z) / ((2 * sigma * sigma) * sigma))
                    voxel[(grid[:, 0] + _x).clip(min=0, max=max_shape[0] - 1),
                          (grid[:, 1] + _y).clip(min=0, max=max_shape[1] - 1),
                          (grid[:, 2] + _z).clip(min=0, max=max_shape[2] - 1)] = gauss
        # 在三维空间内，以每个有点的voxel为中心生成3*3*3大小高斯核kernel，作为GT

        return voxel

    def gaussian2D(self, grid, boxes, sigma_factor=3, device="cuda"):
        box_wh = (boxes[:, -2:] - boxes[:, :2]).cpu().numpy()
        _keep = (box_wh > 0).all(-1)
        boxes = boxes[_keep]
        box_wh = box_wh[_keep]
        for _i, _box in enumerate(boxes):
            w, h = box_wh[_i] // 2
            y = torch.arange(-h, h + 1)[:, None].to(device=device)
            x = torch.arange(-w, w + 1)[None, :].to(device=device)
            sigma_x, sigma_y = (2 * w) / sigma_factor, (2 * h) / sigma_factor
            # generate gaussian-like GT
            gauss = torch.exp(-(x * x + y * y) / (2 * sigma_x * sigma_y))
            h = min(box_wh[_i][1], gauss.shape[0])
            w = min(box_wh[_i][0], gauss.shape[1])
            gauss_area = grid[0, _box[1]:_box[1] + h, _box[0]:_box[0] + w]
            gauss_mask = gauss[:h, :w] > gauss_area[:h, :w]
            gauss_area[gauss_mask] = gauss[:h, :w][gauss_mask]
            grid[0, _box[1]:_box[1] + h, _box[0]:_box[0] + w] = gauss_area

        return grid
