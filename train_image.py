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

def image_classification_test(loader, model, best_avg=0, test_10crop=True):
    start_test = True
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
                #start_time = time.time()
                features, outputs = model(inputs, domain_labels)
                #end_time += (time.time() - start_time)
            else:
                #start_time = time.time()
                features, outputs = model(inputs)
                #end_time += (time.time() - start_time)
            # _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        if  args.dset == 'visda':
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
            # print(best_avg_p)

    return accuracy, best_avg


def train(config):
    ## set pre-process
    logger = SummaryWriter()
    prep_dict = {}
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    prep_config = config["prep"]

    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data = data_config["source"]["batch_size"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=32, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=32, drop_last=True)

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                            transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=32)

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
        ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()

    if args.norm_type == 'rn':
        p = base_network.get_parameters()
        my_p = [v for k,v in base_network.named_parameters() if 'my' in k]
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

    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"]-1:
            base_network.train(False)
            temp_acc, best_avg = image_classification_test(dset_loaders, \
                base_network, best_avg=best_avg_visda)
            temp_model = nn.Sequential(base_network)
            
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
                torch.save(base_network, osp.join(config["output_path"], \
                               "best_model_%s.pth.tar") % config["run_num"])
                torch.save(ad_net, osp.join(config["output_path"], \
                                                  "best_adnet_%s_%s.pth.tar") % (args.CDAN, config["run_num"]))

            if best_avg > best_avg_visda:
                best_avg_visda = best_avg
                torch.save(base_network, osp.join(config["output_path"], \
                               "best_model_%s.pth.tar") % config["run_num"])

            if 'visda' not in args.dset:
                log_str = "iter: {:05d}, precision: {:.5f}, bset_acc:{:.5f}".format(i, temp_acc, best_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(log_str)
            with open(config["record_file"], 'a') as f:
                f.write(log_str + '\n')
        if i % config["snapshot_interval"] == 0:
            torch.save(base_network, osp.join(config["output_path"], \
                               "latest_model_%s.pth.tar") % config["run_num"])
        loss_params = config["loss"]
        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        batch_size = args.batch_size
        inputs = torch.cat((inputs_source,inputs_target),dim=0)
        if args.norm_type == 'dsbn':
            domain_labels = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).long().cuda()
            features, outputs = base_network(inputs, domain_labels)
        else:
            features, outputs = base_network(inputs)
        outputs_source, outputs_target = outputs[:batch_size], outputs[batch_size:]
        softmax_src = nn.Softmax(dim=1)(outputs_source)
        softmax_tgt = nn.Softmax(dim=1)(outputs_target)
        softmax_out = torch.cat((softmax_src, softmax_tgt), dim=0)

        if config['CDAN'] == 'CDAN+E':
            entropy = loss.Entropy(softmax_out)
            # transfer_loss = 0
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)
        elif config['CDAN']  == 'CDAN':
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
        else:
            raise ValueError('Method cannot be recognized.')

        if args.ent:
            ent_loss = -0.1*torch.mean(torch.sum(softmax_tgt*torch.log(softmax_tgt+1e-8),dim=1))
        else:
            ent_loss = 0

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss + ent_loss
        total_loss.backward()
        optimizer.step()
        if my_p is not None:
            for p in my_p:
                p.data.clamp_(min=0.5, max=1)

        if i % config['print_num'] == 0:
            log_str = "iter: {:05d}, classification: {:.5f}, transfer: {:.5f}, method: {:.5f}".format(i, classifier_loss, transfer_loss, ent_loss)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            with open(config["record_file_loss"], 'a') as f:
                f.write(log_str + '\n')
            if config['show']:
                print(log_str)
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

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
    config["record"] = 'record/%s' % args.method
    config["run_num"] = args.run_num
    config["record_file"] = "record/%s/" % args.method + '%s_net_%s_%s_to_%s_num_%s.txt' % (args.method, args.net, args.source, args.target, args.run_num)
    config["record_file_loss"] = "record/%s/" % args.method + '%s_net_%s_%s_to_%s_num_%s_loss.txt' % (args.method, args.net, args.source, args.target, args.run_num)

    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["record"]):
        os.system('mkdir -p ' + config["record"])

    config["prep"] = {"test_10crop":False, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":args.trade_off, "lambda_method":args.lambda_method}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name":network.AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        # exit(0)
        if args.norm_type == 'dsbn':
            config["network"] = {"name":resnetdsbn.resnet50dsbn, \
                "params":{ "use_bottleneck":True, "bottleneck_dim":args.bottle_dim, "new_cls":True} }
        else:
            config["network"] = {"name":network.ResNetFc, \
                "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":args.bottle_dim, "new_cls":True} }
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }

    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }
    if args.dset == 'office-home':
        art_txt = "/home/saulsheng/workspace/project/NewAttention/data/office-home/Art.txt"
        clipart_txt = "/home/saulsheng/workspace/project/NewAttention/data/office-home/Clipart.txt"
        realworld_txt = "/home/saulsheng/workspace/project/NewAttention/data/office-home/Real_World.txt"
        product_txt = "/home/saulsheng/workspace/project/NewAttention/data/office-home/Product.txt"
        if args.source == 'R' : s_dset_path = realworld_txt
        elif args.source == 'C': s_dset_path = clipart_txt
        elif args.source == 'A': s_dset_path = art_txt
        elif args.source == 'P': s_dset_path = product_txt
        if args.target == 'R' : t_dset_path = realworld_txt
        elif args.target == 'C': t_dset_path = clipart_txt
        elif args.target == 'A': t_dset_path = art_txt
        elif args.target == 'P': t_dset_path = product_txt
        if args.multi_source:
            multi_tarA = r'/home/saulsheng/workspace/project/NewAttention/data/office-home/multi_tarA.txt'
            multi_tarC = r'/home/saulsheng/workspace/project/NewAttention/data/office-home/multi_tarC.txt'
            multi_tarP = r'/home/saulsheng/workspace/project/NewAttention/data/office-home/multi_tarP.txt'
            multi_tarR = r'/home/saulsheng/workspace/project/NewAttention/data/office-home/multi_tarR.txt'
            if args.target == 'A':
                s_dset_path = multi_tarA
                t_dset_path = art_txt
            elif args.target == 'C':
                s_dset_path = multi_tarC
                t_dset_path = clipart_txt
            elif args.target == 'P':
                s_dset_path = multi_tarP
                t_dset_path = product_txt
            elif args.target == 'R':
                s_dset_path = multi_tarR
                t_dset_path = realworld_txt
    elif args.dset == 'office31':
        amazon_txt = "/home/saulsheng/workspace/project/NewAttention/data/office/amazon_list.txt"
        dslr_txt = "/home/saulsheng/workspace/project/NewAttention/data/office/dslr_list.txt"
        webcam_txt = "/home/saulsheng/workspace/project/NewAttention/data/office/webcam_list.txt"
        if args.source == 'A': s_dset_path = amazon_txt
        elif args.source == 'D': s_dset_path = dslr_txt
        elif args.source == 'W': s_dset_path = webcam_txt
        if args.target == 'A': t_dset_path = amazon_txt
        elif args.target == 'D': t_dset_path = dslr_txt
        elif args.target == 'W': t_dset_path = webcam_txt
    elif args.dset == 'image-clef':
        p_txt = "/home/saulsheng/workspace/dataset/image-clef/ImageCLEF/pList.txt"
        c_txt = "/home/saulsheng/workspace/dataset/image-clef/ImageCLEF/cList.txt"
        i_txt = "/home/saulsheng/workspace/dataset/image-clef/ImageCLEF/iList.txt"
        if args.source == 'P': s_dset_path = p_txt
        elif args.source == 'C': s_dset_path = c_txt
        elif args.source == 'I': s_dset_path = i_txt
        if args.target == 'P': t_dset_path = p_txt
        elif args.target == 'C': t_dset_path = c_txt
        elif args.target == 'I': t_dset_path = i_txt
    elif args.dset == 'visda':
        s_dset_path = r'/home/saulsheng/workspace/project/NewAttention/data/visda-2017/train_list.txt'
        t_dset_path = r'/home/saulsheng/workspace/project/NewAttention/data/visda-2017/validation_list.txt'
    else:
        s_dset_path = args.s_dset_path
        t_dset_path =args.t_dset_path
    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":s_dset_path, "batch_size":args.batch_size}, \
                      "target":{"list_path":t_dset_path, "batch_size":args.batch_size}, \
                      "test":{"list_path":t_dset_path, "batch_size":args.batch_size}}

    if config["dataset"] == "office31":
        if ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
           ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters       
        config["network"]["params"]["class_num"] = 31 
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')

    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(1,10000)
    print(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


    config["out_file"].write(str(config))
    config["out_file"].flush()
    train(config)
