import glob
import os.path as osp
import os
import random
from PIL import Image

'''
对market1501数据集按照person id进行分类
'''

### 给图像分类并分配图像路径 对应关系 image_class_id:[image_path,image_path]
def get_img_paths(dir_path):
    img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
    # embed()
    class_names = []
    class_paths = {}
    if len(img_paths) > 0:
        for img_path in img_paths:
            # embed()
            class_name = img_path.split('/')[4].split('_')[0]
            class_names.append(class_name)
            class_paths[class_name] = []
            # class_paths[class_name] = img_path.split('data/')[1]
    else:
        print("img_paths contains no path!")

    if len(img_paths) > 0:
        for img_path in img_paths:
            # embed()
            class_name = img_path.split('/')[4].split('_')[0]
            class_names.append(class_name)
            class_paths[class_name].append(img_path.split('data/')[1])
    else:
        print("img_paths contains no path!")

    return class_names,class_paths


### 根据get_img_paths划分好的图像类别和地址,将图像划分为训练集和测试集并保存到本地
def split_train_test(classes):
    for img_class,img_paths in classes.items():
        i = 0

        nums = len(img_paths)
        list = range(1,nums)
        # print("nums",nums)

        sample_size = random.randint(int(nums/3), int(nums/2))
        # 划分为测试集的indexs
        samples = random.sample(list,sample_size)
        for img_path in img_paths:
            i += 1
            img = Image.open(osp.join("../data", img_path))
            # print("img_path:", img_path)

            ### 划分到测试集
            if i in samples:
                path_to_save = osp.join("../data/splited_market1501/test", img_class)
                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)
                    img.save(path_to_save + "/" + str(i) + ".jpg")
                else:
                    img.save(path_to_save + "/" + str(i) + ".jpg")
            ### 划分到训练集
            else:
                path_to_save = osp.join("../data/splited_market1501/train", img_class)
                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)
                    img.save(path_to_save + "/" + str(i) + ".jpg")
                else:
                    img.save(path_to_save + "/" + str(i) + ".jpg")


if __name__ == "__main__":
    # 给对象分类,并分配地址
    class_names,class_paths = get_img_paths("../data/market1501/bounding_box_train")
    # 保存到本地
    split_train_test(class_paths)