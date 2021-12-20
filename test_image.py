import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
from option import args
from sklearn.metrics import fbeta_score, precision_score, recall_score, precision_recall_fscore_support
import network

if args.norm_type == 'dsbn':
    import resnetdsbn
from tensorboardX import SummaryWriter
import time



def image_classification_test(loader, model, best_avg=0):
    start_test = True
    # start_time2 = 0
    end_time = 0
    end_time2 = 0
    times = []
    with torch.no_grad():
        
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            batch_size = inputs.shape[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            domain_labels = torch.from_numpy(np.array([[0]] * batch_size)).long().cuda()
            if args.norm_type == 'dsbn':
                # start_time = time.time()
                features, outputs = model(inputs, domain_labels)
                # end_time += (time.time() - start_time)
                # times.append(time.time() - start_time)

            else:
                # start_time = time.time()
                features, outputs = model(inputs)
                # end_time += (time.time() - start_time)
                # times.append(time.time() - start_time)
            # _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

        _, predict = torch.max(all_output, 1)
        
        if args.dset != 'visda':
            accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

        if args.dset == 'visda':
        # best_avg = 0
            best_per = 0
            gt = all_label.cpu().numpy()
            pred = predict.cpu().numpy()
            labels = np.unique(gt).tolist()
            macro_precision = precision_score(gt, pred, average='macro', labels=labels)
            prec, recall, f1, _ = precision_recall_fscore_support(gt, pred, average=None, labels=labels)
            prec_list = []
            precs = []
            for lab, p, r, f in zip(labels, prec, recall, f1):
                precs.append(p)
                p = '{:d}({:.4f})'.format(int(lab), p)
                prec_list.append(p)

            per_lab_p = 'per label precision: {}'.format(prec_list)
            avg_lab_p = 'avg label precision: {}'.format(np.mean(precs))

            cur_avg = np.mean(precs)
            if cur_avg > best_avg:
                best_avg = cur_avg
                best_per = per_lab_p
            best_avg_p = 'best vag label precision: {}'.format(best_avg)

            print(per_lab_p)
            print(avg_lab_p)
            print(best_avg_p)
            accuracy = 0
         
        
       
    return accuracy, best_avg


def train(config):
    ## set pre-process
    logger = SummaryWriter()

    prep_dict = {}
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    prep_config = config["prep"]

    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data = data_config["source"]["batch_size"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    
    if "webcam" in data_config["source"]["list_path"] or "dslr" in data_config["source"]["list_path"]:
        prep_dict["source"] = prep.image_train31(**config["prep"]['params'])
    else:
        prep_dict["source"] = prep.image_train(**config["prep"]['params'])

    if "webcam" in data_config["target"]["list_path"] or "dslr" in data_config["target"]["list_path"]:
        prep_dict["target"] = prep.image_train31(**config["prep"]['params'])
    else:
        prep_dict["target"] = prep.image_train(**config["prep"]['params'])


    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)

    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                       transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                               shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                  transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                          shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    ## set base network

    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    ## add additional network for some CDANs
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
    if "DANN" in args.CDAN:
        print('DANN')
        ad_net = network.AdversarialNetwork(base_network.output_num(), 512)
    else:
        ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()

    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
    # p = base_network.feature_layers
    # my_p = [v for k,v in p.named_parameters() if 'my' in k]
    if args.norm_type == 'rn':
        p = base_network.get_parameters()
        my_p = [v for k, v in base_network.named_parameters() if 'my' in k]
    else:
        my_p = None
   
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                                         **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    best_avg_visda = 0.0



    root = r'./visda/RN/train2val/'
    path = r'best_model_rn_visda_train2val_resnet50_rn.pth.tar'
    #path = r'best_model_rn_visda_train2val_200911_rn.pth.tar'
    print(path)
    base_network = torch.load(osp.join(root, path))
    print(base_network)
    base_network.train(False)
    temp_acc, best_avg = image_classification_test(dset_loaders, \
                                                   base_network, best_avg=best_avg_visda)

    print('acc:', temp_acc)
    # log_str = "iter: {:05d}, precision: {:.5f}, bset_acc:{:.5f}".format(i, temp_acc, best_acc)
    # print(log_str)

    return temp_acc


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # train config
    config = {}
    config['CDAN'] = args.CDAN
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.num_iterations
    config["print_num"] = args.print_num
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["show"] = args.show
    config["output_path"] = args.dset + '/' + args.output_dir
    config["run_num"] = args.run_num
    config["record_file"] = "record/%s/" % args.method + '%s_net_%s_%s_to_%s_num_%s.txt' % (
    args.method, args.net, args.source, args.target, args.run_num)
    config["record_file_loss"] = "record/%s/" % args.method + '%s_net_%s_%s_to_%s_num_%s_loss.txt' % (
    args.method, args.net, args.source, args.target, args.run_num)


    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop": False, 'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False}}
    config["loss"] = {"trade_off": args.trade_off, "lambda_method": args.lambda_method}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name": network.AlexNetFc, \
                             "params": {"use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}}
    elif "ResNet" in args.net:
        # exit(0)
        if args.norm_type == 'dsbn':
            config["network"] = {"name": resnetdsbn.resnet50dsbn, \
                                 "params": {"use_bottleneck": True, "bottleneck_dim": args.bottle_dim, "new_cls": True}}
        else:
            config["network"] = {"name": network.ResNetFc, \
                                 "params": {"resnet_name": args.net, "use_bottleneck": True,
                                            "bottleneck_dim": args.bottle_dim, "new_cls": True}}
    elif "VGG" in args.net:
        config["network"] = {"name": network.VGGFc, \
                             "params": {"vgg_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    elif "DANN" in args.net or "DANN" in args.CDAN:
        config["network"] = {"name": network.ResNetFc, \
                             "params": {"resnet_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": 0.9, \
                                                               "weight_decay": 0.0005, "nesterov": True},
                           "lr_type": "inv", \
                           "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}}

    if args.dset == 'office-home':
        art_txt = "./data/office-home/Art.txt"
        clipart_txt = "./data/office-home/Clipart.txt"
        realworld_txt = "./data/office-home/Real_World.txt"
        product_txt = "./data/office-home/Product.txt"
        if args.source == 'R' : s_dset_path = realworld_txt
        elif args.source == 'C': s_dset_path = clipart_txt
        elif args.source == 'A': s_dset_path = art_txt
        elif args.source == 'P': s_dset_path = product_txt
        if args.target == 'R' : t_dset_path = realworld_txt
        elif args.target == 'C': t_dset_path = clipart_txt
        elif args.target == 'A': t_dset_path = art_txt
        elif args.target == 'P': t_dset_path = product_txt

    elif args.dset == 'office31':
        amazon_txt = "./data/office/amazon_list.txt"
        dslr_txt = "./data/office/dslr_list.txt"
        webcam_txt = "./data/office/webcam_list.txt"
        if args.source == 'A': s_dset_path = amazon_txt
        elif args.source == 'D': s_dset_path = dslr_txt
        elif args.source == 'W': s_dset_path = webcam_txt
        if args.target == 'A': t_dset_path = amazon_txt
        elif args.target == 'D': t_dset_path = dslr_txt
        elif args.target == 'W': t_dset_path = webcam_txt
    elif args.dset == 'image-clef':
        p_txt = r'./data/image-Clef/pList.txt'
        i_txt = r'./data/image-Clef/iList.txt'
        c_txt = r'./data/image-Clef/cList.txt'
        if args.source == 'I': s_dset_path = i_txt
        elif args.source == 'C': s_dset_path = c_txt
        elif args.source == 'P': s_dset_path = p_txt
        if args.target == 'I': t_dset_path = i_txt
        elif args.target == 'C': t_dset_path = c_txt
        elif args.target == 'P': t_dset_path = p_txt
        print(s_dset_path, t_dset_path)
    elif args.dset == 'visda':
        s_dset_path = r'./data/visda/train_list.txt'
        t_dset_path = r'./data/visda/validation_list.txt'
    else:
        s_dset_path = args.s_dset_path
        t_dset_path =args.t_dset_path
    config["dataset"] = args.dset
    config["data"] = {"source": {"list_path": s_dset_path, "batch_size": args.batch_size}, \
                      "target": {"list_path": t_dset_path, "batch_size": args.batch_size}, \
                      "test": {"list_path": t_dset_path, "batch_size": args.batch_size}}

    if config["dataset"] == "office31":
        if ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003  # optimal parameters
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.0003  # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')

    if args.seed is not None:
        seed = args.seed

    else:
        seed = random.randint(1, 10000)
    print(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # uncommenting the following two lines for reproducing
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    config["out_file"].write(str(config))
    config["out_file"].flush()
    train(config)
