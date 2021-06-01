'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import os
import os.path as osp
import glob

from yolo_base import YOLO

from IPython import embed

yolo = YOLO()

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def crop(img_path,out_path):
    if not os.path.exists("./output"):
        os.makedirs("./output")

    img_name = img_path.split(".png", 1)[0]  # 将目录文件后缀.txt去除
    img_name = os.path.basename(os.path.normpath(img_name))  # normpath规范形式，basename返回文件名,如000242

    image = Image.open(img_path)  # 返回PIL.img对象
    uncroped_image = cv2.imread(img_path)

    objects = yolo.get_boxes(image)
    # print(objects)

    boxes = []
    del_objects = []
    for i, object in enumerate(objects):
        if object[0] != 'person':
            del_objects.append(i)

    for i, del_object in enumerate(del_objects):
        del objects[del_object - i]

    # print(objects)
    for object in objects:
        box = []
        box.append(object[2])
        box.append(object[3])
        box.append(object[4])
        box.append(object[5])
        boxes.append(box)

    # print(boxes)
    r_image = yolo.draw_boxes(image, objects)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    elif os.path.exists(out_path):
        del_file(out_path)

    for i in range(len(boxes)):
        # top, left, bottom, right = boxes[i]
        # 或者用下面这句等价
        top = boxes[i][0]
        left = boxes[i][1]
        bottom = boxes[i][2]
        right = boxes[i][3]

        top = top - 5
        left = left - 5
        bottom = bottom + 5
        right = right + 5

        # 左上角点的坐标
        top = int(max(0, np.floor(top + 0.5).astype('int32')))

        left = int(max(0, np.floor(left + 0.5).astype('int32')))
        # 右下角点的坐标
        bottom = int(min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32')))
        right = int(min(np.shape(image)[1], np.floor(right + 0.5).astype('int32')))

        # embed()

        # 问题出在这里：不能用这个方法，看两个参数是长和宽，是从图像的原点开始裁剪的，这样肯定是不对的
        croped_region = uncroped_image[top:bottom, left:right]  # 先高后宽
        croped_region = cv2.resize(croped_region, (64, 128))
        # 将裁剪好的目标保存到本地

        cv2.imwrite(out_path + str(i+1) + ".png", croped_region)

    # r_image.show()

### 2 views
def get_cropped_pics(input_base_path,output_base_path):
    # dir_path = './input'
    img_paths = glob.glob(osp.join(input_base_path, '*.jpg'))
    i = 0
    for img_path in img_paths:
        # embed()
        if i > 1:
            break
        i += 1
        output_path = osp.join(output_base_path,"/",img_path.split('./')[1].split("\\")[1].split(".")[0],"/pics/")
        img_path.split()
        crop(img_path,output_path)

### 8 views
def get_cropped_8pics(input_base_path,output_base_path):
    # dir_path = './input'
    img_paths = glob.glob(osp.join(input_base_path, '*.jpg'))
    i = 0
    for img_path in img_paths:
        # embed()
        if i > 7:
            break
        i += 1
        output_path = osp.join(output_base_path,"/",img_path.split('./')[1].split("\\")[1].split(".")[0],"/pics/")
        img_path.split()
        crop(img_path,output_path)


# if __name__ == "__main__":
#     img_path = './input/view52.jpg'
#     out_path = "./data/view2/pics"
#     # img_path2 = './img/view52.jpg'
#     crop(img_path,out_path)
#     # crop(img_path2)
# embed()
# r_image.show()
# r_image.save("./output/detected_img_view2.jpg",quality=95)
# Image.save("./output/detected_img_view2.jpg",r_image)
# cv2.imread("./output/detected_img_view2.jpg",r_image)