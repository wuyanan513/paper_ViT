
import json
import os
import cv2
import numpy as np
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torchvision import transforms
from sklearn.metrics import roc_curve, auc
from models.model import ViT
from util import MyDataset
from uitil import progress_bar
import pandas as pd

from torch.utils.data import DataLoader
from confusionmatrix import ConfusionMatrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm as tqdm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch emphysema Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4?
    parser.add_argument('--bs', type=int,default='2')
    parser.add_argument('--n_epochs', type=int, default='50')
    parser.add_argument('--pretrained',type=str, default='True')
    parser.add_argument('--model_name', type=str,default='B_16_imagenet1k')  
    parser.add_argument('--seed', type=int, default='50')



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention
def show_cam_on_image(img, mask):
    img = img.squeeze().transpose(1,2,0)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = cv2.resize(heatmap,(384,384))
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam
def test(testloader,device,net):
    # global best_acc 
    # best_acc=0
    json_label_path = 'class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=4, labels=labels)
    net.eval()
    # test_loss = 0
    # correct = 0
    # total = 0
    test_score = []
    score_list = []     # 存储预测得分
    label_list = []
    with torch.no_grad():
        test_total=0
        test_correct=0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            score_tmp = outputs.detach().cpu().numpy()
            score_list.extend(score_tmp[0,targets.detach().cpu().numpy()])     # 存储预测得分
            label_list.extend(targets.detach().cpu().numpy())
            loss = criterion(outputs, targets)
            output = torch.softmax(outputs, dim=1)
            output = torch.argmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            test_acc = test_correct/test_total
            test_score.append(test_acc)
            confusion.update(output.to("cpu").numpy(),targets.to("cpu").numpy())
        print(f'test_accuracy:{np.mean(test_score)}')
            # print(inputs.shape)
            # print(net)
            # summary(net,(3,60,60))
         # 混淆矩阵
        confusion.plot()
        confusion.summary() 
        data_test = {'label':label_list,'score':score_list}
        df_test=pd.DataFrame(data_test)
        df_test.to_csv('Result_test_pretrained_ViT.csv',index=None)
        print('Finished saving csvfile!')    
if __name__ == '__main__':
    args = get_args()
    setup_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Data
    print('==> Preparing data..')
    train_txt_path = os.path.join("data_post\data", "train.txt")#classification_
    valid_txt_path = os.path.join("data_post\data", "valid.txt")
    # test_txt_path = os.path.join("data", "test.txt")
    #数据预处理

    normMean = [0.18900262, 0.18900262, 0.18900262]
    normStd = [0.19090985, 0.19090985, 0.19090985]
    normTransform = transforms.Normalize(normMean, normStd)
    trainTransform = transforms.Compose([
    # transforms.Resize(32),
    #  transforms.RandomCrop(32, padding=4),
        transforms.Resize(384),
        # transforms.RandomCrop(60, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor(),
        normTransform
        ]) 

    validTransform = transforms.Compose([
        transforms.Resize(384),
        transforms.ToTensor(),
        normTransform
        ])
    testTransform = transforms.Compose([
        transforms.Resize(384),
        transforms.ToTensor(),
        normTransform
        ])

     # 构建MyDataset实例
    train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
    valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)
    # test_data = MyDataset(txt_path=test_txt_path, transform=testTransform)

    # 构建DataLoder
    trainloader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True)
    validloader = DataLoader(dataset=valid_data, batch_size=args.bs)
    # testloader = DataLoader(dataset=test_data, batch_size=1)
    # 构建MyDataset实例
    # train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
    # valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)

    # # 构建DataLoder
    # trainloader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
    # testloader = DataLoader(dataset=valid_data, batch_size=2)

    # Load model
    net = ViT(args.model_name, pretrained=args.pretrained,image_size=384,num_classes=4)
    # print (net)
    # net = nn.DataParallel(net)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    # reduce LR on Plateau
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)

#### Training
    total_train_loss = []
    total_train_score = []
    total_val_loss = []
    total_val_score = []
    for epoch in range(1, args.n_epochs+1):
        net.train()
        train_loss = []
        train_score = []
        valid_loss = []
        valid_score = []
        correct = 0
        total = 0
        val_correct = 0
        val_total = 0
        pbar = tqdm(trainloader, desc = 'description')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            # print(net)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            # train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = correct/total
            train_loss.append(loss_value)
            train_score.append(acc)
            pbar.set_description(f"Epoch: {epoch}, loss: {loss_value}, ACC: {acc}")

##### Validation
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(validloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                score_tmp = outputs
                loss = criterion(outputs, targets)
                output = torch.softmax(outputs, dim=1)
                output = torch.argmax(outputs, dim=1)
                # confusion.update(output.to("cpu").numpy(), targets.to("cpu").numpy())
                test_loss = loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                test_loss = loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                val_acc = val_correct/val_total
                valid_loss.append(test_loss)
                valid_score.append(val_acc)
            if np.mean(val_acc)>0.92:
                torch.save(net.state_dict(), f'model_weights{np.mean(val_acc)}.pth')

        total_train_loss.append(np.mean(train_loss))
        total_train_score.append(np.mean(train_score))
        total_val_loss.append(np.mean(valid_loss))
        total_val_score.append(np.mean(valid_score))
    # plt.figure(0)
    #
    # plt.figure(figsize=(15,5))
    # plt.subplot(1, 2, 1)
    # # sns.set_style(style="darkgrid")
    #
    # sns.lineplot(x=range(1,epoch+1), y=total_train_loss, label="Train")
    # sns.lineplot(x=range(1,epoch+1), y=total_val_loss, label="Valid")
    # plt.legend(prop={ 'size': 14})
    # # plt.title("Loss")
    # plt.xlabel("Epoch(-)",fontsize=14)
    # plt.ylabel("Loss",fontsize=14)
    # plt.yticks(fontproperties = 'Times New Roman', size = 14)
    # plt.xticks(fontproperties = 'Times New Roman', size = 14)
    #
    # plt.subplot(1, 2, 2)
    # # sns.set_style(style="darkgrid")
    # sns.lineplot(x=range(1,epoch+1), y=total_train_score, label="Training")
    # sns.lineplot(x=range(1,epoch+1), y=total_val_score, label="Valid")
    # plt.legend(prop={ 'size': 14})
    # # plt.title("Accucary")
    # plt.xlabel("Epoch(-)",fontsize=14)
    # plt.ylabel("Accucary",fontsize=14)
    # plt.yticks(fontproperties = 'Times New Roman', size = 14)
    # plt.xticks(fontproperties = 'Times New Roman', size = 14)
    # plt.savefig("1.png")
    # plt.show()
    # test(testloader,device,net)  