# -------------------------------------#
#       创建YOLO类
# -------------------------------------#
import json
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
from nets.yolo4 import YoloBody
import torch.backends.cudnn as cudnn
from PIL import Image, ImageFont, ImageDraw
from torch.autograd import Variable
from utils.utils import non_max_suppression, bbox_iou, DecodeBox, letterbox_image, yolo_correct_boxes


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo4_weights.pth',
        "anchors_path": 'model_data/coco_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "model_image_size": (608, 608, 3),
        "confidence": 0.6,
        "iou": 0.5,
        "cuda": False,
        "cut_iou": 0.1
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   获得所有的先验框
    # ---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):

        self.net = YoloBody(len(self.anchors[0]), len(self.class_names)).eval()

        # 加快模型训练的效率
        # print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        # print('Finished!')

        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(
                DecodeBox(self.anchors[i], len(self.class_names), (self.model_image_size[1], self.model_image_size[0])))

        # print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))


    def get_boxes(self, image):
        image = image.convert('RGB')
        image_shape = np.array(np.shape(image)[0:2])  # 得到图像的长和宽

        crop_img = np.array(letterbox_image(image, (self.model_image_size[0], self.model_image_size[1])))
        photo = np.array(crop_img, dtype=np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)

        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, 1)
        batch_detections = non_max_suppression(output, len(self.class_names),
                                               conf_thres=self.confidence,
                                               nms_thres=self.iou)
        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return 'noobj'

        top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
        top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
        top_label = np.array(batch_detections[top_index, -1], np.int32)
        top_bboxes = np.array(batch_detections[top_index, :4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1],
                                                                                                      -1), np.expand_dims(
            top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

        # 去掉灰条
        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                   np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)



        objects = []
        count = 0

        for i, c in enumerate(top_label):
            object = []
            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            object.append(predicted_class)
            object.append(score)
            object.append(top)
            object.append(left)
            object.append(bottom)
            object.append(right)
            objects.append(object)
            count += 1


        return objects



    def get_iou_score(self, obj1, obj2):
        if (obj1[0] != obj2[0]):
            return 0

        left_max = max(obj1[3], obj2[3])
        top_max = max(obj1[2], obj2[2])
        right_min = min(obj1[5], obj2[5])
        bottom_min = min(obj1[4], obj2[4])
        # 两矩形相交时计算IoU
        if (left_max < right_min and bottom_min > top_max):  # 判断有无重叠部分
            rect1_area = (obj1[5] - obj1[3]) * (obj1[4] - obj1[2])
            rect2_area = (obj2[5] - obj2[3]) * (obj2[4] - obj2[2])
            area_cross = (bottom_min - top_max) * (right_min - left_max)
            return area_cross / (rect1_area + rect2_area - area_cross)
        else:
            return 0


    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        image = image.convert('RGB')
        image_shape = np.array(np.shape(image)[0:2])  # 得到图像的长和宽

        crop_img = np.array(letterbox_image(image, (self.model_image_size[0], self.model_image_size[1])))
        photo = np.array(crop_img, dtype=np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)

        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, 1)
        batch_detections = non_max_suppression(output, len(self.class_names),
                                               conf_thres=self.confidence,
                                               nms_thres=self.iou)
        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image

        top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
        top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
        top_label = np.array(batch_detections[top_index, -1], np.int32)
        top_bboxes = np.array(batch_detections[top_index, :4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1],
                                                                                                      -1), np.expand_dims(
            top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

        # 去掉灰条
        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                   np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[
            0]  # 输入图像shape和模型shape的比较，用于调整预测框厚度

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            if thickness == 0:  # 输入图像大小小于608
                for i in range(2):
                    draw.rectangle(  # 画预测框
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[self.class_names.index(predicted_class)])

            else:
                for i in range(thickness):
                    draw.rectangle(  # 画预测框
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[self.class_names.index(predicted_class)])

            draw.rectangle(  # 画class框
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
            # print(right - left, bottom - top)

        return image

    def draw_boxes(self, image, objects):
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[
            0]  # 输入图像shape和模型shape的比较，用于调整预测框厚度

        for i in range(len(objects)):
            label = '{} {:.2f}'.format(objects[i][0], objects[i][1])

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label)

            if objects[i][2] - label_size[1] >= 0:
                text_origin = np.array([objects[i][3], objects[i][2] - label_size[1]])
            else:
                text_origin = np.array([objects[i][3], objects[i][2] + 1])

            if thickness == 0:  # 输入图像大小小于608
                for j in range(2):
                    draw.rectangle(  # 画预测框
                        [objects[i][3] + j, objects[i][2] + j, objects[i][5] - j, objects[i][4] - j],
                        outline=self.colors[self.class_names.index(objects[i][0])])

            else:
                for j in range(thickness):
                    draw.rectangle(  # 画预测框
                        [objects[i][3] + j, objects[i][2] + j, objects[i][5] - j, objects[i][4] - j],
                        outline=self.colors[self.class_names.index(objects[i][0])])

            draw.rectangle(  # 画class框
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(objects[i][0])])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def file_lines_to_list(self, path):
        # open txt file lines to a list
        with open(path) as f:
            content = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        return content

    def detect_muti_model(self, txt, image):
        lines = self.file_lines_to_list(txt)
        th = 0.9
        dict = {}
        objects = []
        for line in lines:
            cls, num = line.split('=')
            dict[cls] = int(num)

        objs = self.get_boxes(image)
        if objs == 'noobj':
            return image
        objects.sort(reverse=True)

        for obj in objs:
            # print(obj)
            if dict.get(obj[0], 'none') == 'none':
                objects.append(obj)
            else:
                if dict[obj[0]] != 0:
                    dict[obj[0]] -= 1
                    objects.append(obj)
                else:
                    if obj[1] > th:
                        objects.append(obj)

        # print(objects)

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[
            0]  # 输入图像shape和模型shape的比较，用于调整预测框厚度

        for i in range(len(objects)):
            label = '{} {:.2f}'.format(objects[i][0], objects[i][1])

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label)

            if objects[i][2] - label_size[1] >= 0:
                text_origin = np.array([objects[i][3], objects[i][2] - label_size[1]])
            else:
                text_origin = np.array([objects[i][3], objects[i][2] + 1])

            if thickness == 0:  # 输入图像大小小于608
                for j in range(2):
                    draw.rectangle(  # 画预测框
                        [objects[i][3] + j, objects[i][2] + j, objects[i][5] - j, objects[i][4] - j],
                        outline=self.colors[self.class_names.index(objects[i][0])])

            else:
                for j in range(thickness):
                    draw.rectangle(  # 画预测框
                        [objects[i][3] + j, objects[i][2] + j, objects[i][5] - j, objects[i][4] - j],
                        outline=self.colors[self.class_names.index(objects[i][0])])

            draw.rectangle(  # 画class框
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(objects[i][0])])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image





















