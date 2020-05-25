from patient import Patient
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from lhyp_dataset import LhypDataset
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import copy
from sklearn.metrics import f1_score, confusion_matrix
from PIL import Image
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import datetime


def training(model_name, num_of_classes, batch_size, num_of_epochs, learning_rate):


    trans = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.RandomRotation(25, fill=(0,)),
        transforms.RandomCrop((230, 230), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    val_trans = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    with open('../data/pickle/train'+num_of_classes, 'rb') as infile:
        train_pat = pickle.load(infile, encoding='bytes')
    with open('../data/pickle/val'+num_of_classes, 'rb') as infile:
        val_pat = pickle.load(infile, encoding='bytes')
    with open('../data/pickle/test'+num_of_classes, 'rb') as infile:
        test_pat = pickle.load(infile, encoding='bytes')

    class_sample_count = [0]*num_of_classes
    for sample in train_pat:
        class_sample_count[sample['label']] += 1
    print(class_sample_count)

    weights = []
    for data in train_pat:
        weights.append(1/class_sample_count[data['label']])
    sampler = WeightedRandomSampler(weights, len(weights))

    dataset = LhypDataset(train_pat, trans)

    val_dataset = LhypDataset(
        val_pat, val_trans)

    test_dataset = LhypDataset(
        test_pat, val_trans)

    print(len(dataset), len(val_dataset), len(test_dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    if model_name == 'densenet':
        model = torchvision.models.densenet121(
            pretrained=True, progress=True, drop_rate=0.2)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(
            num_features, num_of_classes))
    elif model_name == 'resnet':
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = nn.Sequential(nn.Dropout(0.65), nn.Linear(
            model.fc.in_features, num_of_classes))
    elif model_name == 'squeezenet':
        model = torchvision.models.squeezenet1_1(
            pretrained=True, progress=True)
        model.classifier[1] = nn.Conv2d(
            512, num_of_classes, kernel_size=(1, 1), stride=(1, 1))

    # model.load_state_dict(torch.load('models/mobile_3cat.pth'))
    best_model_wts = copy.deepcopy(model.state_dict())

    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, num_of_epochs)

    # optimizer = optim.SGD(
    #     model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-3)

    min_loss = 9999999999999999999.0

    writer = SummaryWriter('drive/My Drive/onlab/LHYP/runs/2')

    for iteration in range(num_of_epochs):
        print(iteration)
        #
        # Training
        #
        train_loss, train_acc = train(model, dataloader, criterion, optimizer)

        writer.add_scalar('Training loss', train_loss, iteration)
        writer.add_scalar('Training accuracy', train_acc, iteration)
        print('Train loss: ', train_loss, ' Train acc ', train_acc)
        #
        # Validating
        #
        val_running_loss, val_acc = validate(model, val_dataloader, criterion)

        writer.add_scalar('Validation loss', val_running_loss, iteration)
        writer.add_scalar('Validation accuracy',
                          val_acc, iteration)

        if val_running_loss < min_loss:
            min_loss = val_running_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print('Val loss: ', val_running_loss,
              ' Val acc ', val_acc)

        scheduler.step()

    # Testing the model
    #torch.save(best_model_wts, 'models/mobile_3cat.pth')
    model.load_state_dict(best_model_wts)
    conf_mat = test(model, test_dataloader, criterion)

    if num_of_classes == 2:
        classes = {0: 'Normal', 1: 'Other'}
    elif num_of_classes == 3:
        classes = {0: 'Normal', 1: 'HCM', 2: 'Other'}

    df_cm = pd.DataFrame(conf_mat, index=[i for i in classes.values()],
                         columns=[i for i in classes.values()])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('../charts/'+ model_name +'_'+num_of_classes +'_chart' +
                datetime.datetime.now().strftime("%H:%M:%S") + '.png')


def train(model, dataloader, criterion, optimizer):
    running_loss = 0
    correct = 0.00
    total = 0.00

    model.train()
    for i, input_datas in enumerate(dataloader, 0):
        datas, labels = input_datas

        if torch.cuda.is_available():
            datas = datas.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        outputs = model(datas)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss, correct/total


def validate(model, val_dataloader, criterion):
    val_running_loss = 0
    val_correct = 0.00
    val_total = 0.00

    model.eval()
    for i, val_input_datas in enumerate(val_dataloader, 0):
        val_datas, val_labels = val_input_datas

        if torch.cuda.is_available():
            val_datas = val_datas.cuda()
            val_labels = val_labels.cuda()

        val_outputs = model(val_datas)
        val_loss = criterion(val_outputs, val_labels)

        val_running_loss += val_loss.item()
        _, val_predicted = torch.max(val_outputs, 1)
        val_total += val_labels.size(0)
        val_correct += val_predicted.eq(val_labels).sum().item()

    return val_running_loss, val_correct/val_total


def test(model, test_dataloader, criterion):
    val_running_loss = 0
    val_correct = 0.00
    val_total = 0.00
    y_true = np.array([])
    y_pred = np.array([])
    model = model.cpu()
    model.eval()

    for i, test_input_datas in enumerate(test_dataloader, 0):
        val_datas, val_labels = test_input_datas
        y_true = np.concatenate((y_true, val_labels.numpy()), 0)
        val_outputs = model(val_datas)
        val_loss = criterion(val_outputs, val_labels)

        val_running_loss += val_loss.item()
        _, val_predicted = torch.max(val_outputs, 1)

        y_pred = np.concatenate((y_pred, val_predicted.numpy()), 0)
        val_total += val_labels.size(0)
        val_correct += val_predicted.eq(val_labels).sum().item()

    print('Test loss: ', val_running_loss, ' Test acc ', val_correct/val_total)
    print('F1 score:    ', f1_score(y_true, y_pred, average='micro'))
    return confusion_matrix(y_true, y_pred, normalize='pred')


def main():
    training('resnet', 2, 32, 10, 6e-5)


if __name__ == "__main__":
    main()