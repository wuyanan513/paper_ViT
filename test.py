
import torch
import torch.nn as nn
import numpy as np
import os
from timm.models.vision_transformer import VisionTransformer
import torchvision.transforms as transforms
from util import MyDataset
from torch.utils.data import DataLoader
import json
from pytorch_pretrained_vit.model import ViT
from confusionmatrix import ConfusionMatrix
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_txt_path = os.path.join(r"data_post\public_data", "train.txt")
PATH = 'model_weights0.9592476489028213.pth'

normMean = [0.21542308, 0.21542308, 0.21542308]
normStd = [0.20693116, 0.20693116, 0.20693116]
normTransform = transforms.Normalize(normMean, normStd)
testTransform = transforms.Compose([
# transforms.Resize(32),
#  transforms.RandomCrop(32, padding=4),
    transforms.Resize(384),
    # transforms.RandomCrop(60, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.5),
    transforms.ToTensor(),
    normTransform
    ]) 
test_data = MyDataset(txt_path=test_txt_path, transform=testTransform)
# 构建DataLoder
testloader = DataLoader(dataset=test_data, batch_size=1)
model_name = 'B_16_imagenet1k'

net = ViT(model_name, pretrained=True, image_size=384, num_classes=4)
# print (net)
# net = nn.DataParallel(net)
net = net.to(device)
# net = VisionTransformer(
#         img_size = 60,
#         patch_size = 4,
#         num_classes = 4,
#         embed_dim=128,
#         depth=4,
#         num_heads=8,
#         mlp_ratio=4)
net.load_state_dict(torch.load(PATH))
# net.load_state_dict()
net.to(device)

json_label_path = 'class_indices.json'
assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
json_file = open(json_label_path, 'r')
class_indict = json.load(json_file)

labels = [label for _, label in class_indict.items()]
confusion = ConfusionMatrix(num_classes=4, labels=labels)
net.eval()
test_loss = 0
correct = 0
total = 0
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        output = torch.softmax(outputs, dim=1)
        output = torch.argmax(outputs, dim=1)
        confusion.update(output.to("cpu").numpy(), targets.to("cpu").numpy())
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Acc: %.3f', 100.*correct/total)
    # confusion.plot()
    confusion.summary()
    confusuionmatrix = confusion.plot() 