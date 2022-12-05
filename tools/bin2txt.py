
import numpy as np
def bin2txt(ori_path):
    # data = np.fromfile(ori_path, np.float32).reshape(-1, 3)
    data = np.fromfile(ori_path, np.float32).reshape(-1, 7)
    # import pdb
    # pdb.set_trace()
    data = np.delete(data, np.where(data[:, 2] == -10000), axis=0)
    # np.savetxt((ori_path.split(".")[0] + '.txt'),data,fmt="%.4f",delimiter=",")
    np.savetxt(("/home/zhanghaoming/visual/02018" + '.txt'),data,fmt="%.4f",delimiter=" ")
 
bin2txt("/home/baohao/chenhaifeng/data/vod/radar_5frames/training/velodyne/02018.bin")
