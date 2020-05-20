import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import torch.optim as optim
import torchvision.models as models
import re
from glob import glob
from PIL import Image
import sys
import time
import torchvision.transforms as transforms

seed = 20

torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed)
class StudentNet(nn.Module):

    def __init__(self, base=16, width_mult=1):
        super(StudentNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.cnn = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 2),
            conv_dw(256, 256, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 11),
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folderName, transform=None):
        self.transform = transform
        self.data = []
        self.label = []

        for img_path in sorted(glob(folderName + '/*.jpg')):
            try:
                # Get classIdx by parsing image path
                class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
            except:
                # if inference mode (there's no answer), class_idx default 0
                class_idx = 0

            image = Image.open(img_path)
            # Get File Descriptor
            image_fp = image.fp
            image.load()
            # Close File Descriptor (or it'll reach OPEN_MAX)
            image_fp.close()

            self.data.append(image)
            self.label.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]


trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

def get_dataloader(mode='training', batch_size=16):

    assert mode in ['training', 'testing', 'validation']

    dataset = MyDataset(
        f'./food-11/{mode}',
        transform=trainTransform if mode == 'training' else testTransform)
    print(dataset.__len__())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    return dataloader

def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 讓logits的log_softmax對目標機率(teacher的logits/T後softmax)做KL Divergence。
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss
# get dataloader
print("Loading training data...")
train_dataloader = get_dataloader('training', batch_size=32)
print("Loading validation...")
valid_dataloader = get_dataloader('validation', batch_size=32)

print("Loading teacher net...")
teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
student_net = StudentNet(base=16).cuda()

teacher_net.load_state_dict(torch.load(f'./teacher_resnet18.bin'))
print("finish loading...")
optimizer = optim.Adam(student_net.parameters(), lr=1e-3)
optimizer1 = optim.Adam(teacher_net.parameters(), lr=1e-3)
def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空 optimizer
        optimizer.zero_grad()
        optimizer1.zero_grad()
        # 處理 input
        inputs, hard_labels = batch_data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        # 因為Teacher沒有要backprop，所以我們使用torch.no_grad
        # 告訴torch不要暫存中間值(去做backprop)以浪費記憶體空間。
        
        soft_labels = teacher_net(inputs)

        if update:
            logits = student_net(inputs)
            # 使用我們之前所寫的融合soft label&hard label的loss。
            # T=20是原始論文的參數設定。
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 30, alpha) + loss_fn_kd(soft_labels, hard_labels, logits, 30, 0.3)
            loss.backward()
            optimizer.step()    
            optimizer1.step()
        else:
            # 只是算validation acc的話，就開no_grad節省空間。
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 30, alpha)
            
        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num


# TeacherNet永遠都是Eval mode.
#teacher_net.eval()
now_best_acc = 0
num_epoch = 250
print("start training...")
for epoch in range(num_epoch):
    if epoch == 150:
        optimizer = optim.SGD(student_net.parameters(), lr = 2e-4, momentum=0.9)
        optimizer1 = optim.SGD(teacher_net.parameters(), lr = 2e-4, momentum=0.9)
    start_time = time.time()
    student_net.train()
    train_loss, train_acc = run_epoch(train_dataloader, update=True,alpha = 0.5)
    student_net.eval()
    valid_loss, valid_acc = run_epoch(valid_dataloader, update=False,alpha = 0.5)

    # 存下最好的model。
    if valid_acc > now_best_acc:
        now_best_acc = valid_acc
        torch.save(student_net.state_dict(), sys.argv[1])
    print('run time: {:6.4f}, epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
        time.time()-start_time, epoch, train_loss, train_acc, valid_loss, valid_acc))




























