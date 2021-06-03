import glob
import os.path as osp
import os
import random
from PIL import Image

'''
对mars数据集按照person id进行分类
'''

### 先把所有的图像分类别保存
### 具体方式为：给图像分类并分配图像路径 对应关系 image_class_id:[image_path,image_path]
def get_img_paths(dir_path):
    # embed()
    img_paths = glob.glob(osp.join(dir_path,"*","*"))

    print("img_paths",img_paths)
    print("img_paths length:",len(img_paths))
    class_labels = []
    i = 0
    for img_path in img_paths:
        i += 1
        class_label = img_path.split("/")[7]
        if class_label not in class_labels:
            class_labels.append(class_label)

        sub_img_paths = glob.glob(osp.join(img_path,"*.jpg"))
        # embed()
        print("sub_img_paths[0]",sub_img_paths[0])
        img = Image.open(sub_img_paths[0])
        save_to_path = "/home/gaoziqiang/resource/dataset/split_mars/mars_train/" + class_label
        if not osp.exists(save_to_path):
            os.makedirs(save_to_path)
            img.save(save_to_path + "/" + str(i) + ".jpg")
        else:
            img.save(save_to_path + "/" + str(i) + ".jpg")
    print("====>end")
    print("class nums:",len(class_labels))


### 然后根据get_img_paths划分好的图像类别和地址,将图像划分为训练集和测试集并保存到本地
def split_train_test(dir_path):
    base_dirs  = glob.glob(osp.join(dir_path,"*"))
    print("base_dirs:", base_dirs)
    print("base_dirs nums:", len(base_dirs))
    for base_dir in base_dirs:
        img_paths = glob.glob(osp.join(base_dir,"*.jpg"))

        i = 0

        nums = len(img_paths)
        list = range(1, nums)
        print("nums:", nums)

        sample_size = random.randint(int(nums / 3), int(nums / 2))
        # 划分为测试集的indexs
        samples = random.sample(list, sample_size)
        for img_path in img_paths:
            i += 1
            img_class = img_path.split('/')[7]
            img = Image.open(img_path)
            # embed()
            # print("img_path:", img_path)

            ### 划分到测试集
            if i in samples:
                path_to_save = osp.join("/home/gaoziqiang/resource/dataset/split_mars/test", img_class)
                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)
                    img.save(path_to_save + "/" + str(i) + ".jpg")
                else:
                    img.save(path_to_save + "/" + str(i) + ".jpg")
            ### 划分到训练集
            else:
                path_to_save = osp.join("/home/gaoziqiang/resource/dataset/split_mars/train", img_class)
                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)
                    img.save(path_to_save + "/" + str(i) + ".jpg")
                else:
                    img.save(path_to_save + "/" + str(i) + ".jpg")


if __name__ == "__main__":
    ### 先把所有的图像分类别保存 这里dir_path使用绝对路径
    total_img_paths = glob.glob(osp.join("/home/gaoziqiang/resource/dataset/mars/train_split","0*","0*"))
    get_img_paths("/home/gaoziqiang/resource/dataset/mars/train_split")
    ### 然后划分训练集和测试集
    split_train_test("/home/gaoziqiang/resource/dataset/split_mars/mars_train")