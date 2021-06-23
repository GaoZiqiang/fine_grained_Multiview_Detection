import glob
import numpy as np
import random
import os.path as osp
from PIL import Image

from IPython import embed

def restructure(base_path):
    base_paths = glob.glob(osp.join(base_path,'*'))
    print("base_paths length：",len(base_paths))
    senus = list(range(1,5000))
    random_senus = random.sample(senus,4000)
    print("random_senus:",random_senus)
    i = 0
    j = 1
    for base_path in base_paths:
        print("base_path i:",j)
        j += 1
        img_paths = glob.glob(osp.join(base_path,'*.jpg'))
        for img_path in img_paths:
            i += 1
            # embed()
            image = Image.open(img_path)
            random_senu = random_senus[i]
            result_dir = './data/result/200/data/'
            image.save(result_dir + str(random_senu) + '.jpg')
            # embed()
    print("total i:",i)

if __name__ == "__main__":
    base_path = '/media/gaoziqiang/244D-0C54/datasets/200/data'
    restructure(base_path)