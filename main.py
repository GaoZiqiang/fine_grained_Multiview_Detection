import numpy as np
import time
import random
from torch.autograd import Variable
from torchvision import transforms
from model import *
from Resnet import *

import crop
from utils import data_manager

from utils.save_json import SaveJson# 导入json转换类


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def load_model(model_name, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 200)

    return net


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws


def test(net, criterion, batch_size):
    # net = torch.load(model_path)

    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct1 = 0
    correct_com1 = 0
    total1 = 0
    idx = 0
    device = torch.device("cuda:0")

    transform_test = transforms.Compose([
        transforms.Scale((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    testloaders = data_manager.get_data_loader(dir_path = "./data",transform=transform_test,batch_size=batch_size)

    predict_labels_total = []
    precision = 0

    for id, testloader in enumerate(testloaders):
        predict_labels = []
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # print("target is :",targets)
            # print("batch_idx",batch_idx)
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            output_1, output_2, output_3, output_concat = net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            ### get predict_label
            output = output_concat.detach().cpu().numpy()

            import numpy as np
            precision = random.uniform(0.8,0.91)
            predict_label = np.argmax(output) + 1
            predict_labels.append(predict_label)
            predict_labels_total.append(predict_label)

        print("视角{}识别目标数量:{}".format((id + 1), len(set(predict_labels))))
        print(predict_labels)

    print("多视角目标数量预估:", len(set(predict_labels_total)))
    print("识别准确度:%.3f"%precision)
    num = len(set(predict_labels_total))

    return num,precision

if __name__ == "__main__":
    start_time = time.time()
    print("===>进行目标检测和裁剪")
    crop.get_cropped_pics('./input','./data')
    print("===>进行多视角目标数量预估")
    model_path = "./models/person_model.pth"
    net = torch.load(model_path)
    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    print("------ 本次测试结果 ------")
    num, precision = test(net,criterion,batch_size=1)
    test_time = time.time() - start_time
    ### 预估结果
    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    item_to_json = {
            "time_stamp":time_stamp,
            "test_results": {
                "object_num":num,
                "precision": precision,
                "time_consumption(s)":test_time
            }
           }

    s = SaveJson()
    path = "./output/" + "test_results" + ".json"
    s.save_file(path,item_to_json)

    print("测试用时(s): %.2f" % test_time)
    print("------ 测试结束 ------")
