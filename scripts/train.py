import torch
import torch.nn as nn
import argparse
import json
import sys
import cv2

sys.path.insert(0, "../")

from utils import transform
from utils import dataset
from utils import config
from model.pspnet import PSPNet

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='../configs/cityscapes/cityscapes_pspnet50_sat.yaml', help='config file')
    parser.add_argument('--attack_iters', default=7, type=int)
    parser.add_argument('--eps', default=0.03, type=float)
    parser.add_argument('--alpha', default=0.01, type=float)
    parser.add_argument('--rand_init', default=None, type=str, help='norm, uniform')

    parser.add_argument('--data_adv', default='mixed', type=str, help='mixed, adv_only, clean_only')
    parser.add_argument('--method_name', default='', type=str)
    parser.add_argument('--num_batch', default=0, type=int)
    parser.add_argument('--manual_seed', default=1, type=int)

    # Setting for Attack method
    parser.add_argument('--loss', default='ce', type=str)
    parser.add_argument('--miscls_loss_lamb', default=0.5, type=float)
    parser.add_argument('opts', help='see configs/cityscapes/cityscapes_pspnet50_sat.yamll for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

# Extract the hyperparameters and device configuration
args = get_parser()
args.save_path = args.save_path + '_' + args.method_name

# Define your loss function
loss_fn = nn.CrossEntropyLoss()

def train(model, dataloader, optimizer):
    for batch_idx, (data, target) in enumerate(dataloader):
        # Move the data and target tensors to the device
        data = data.to(device)
        target = target.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculate the loss
        loss = loss_fn(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_accuracy += (predicted == target).sum().item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_accuracy / len(dataloader.dataset)

    return avg_loss, accuracy

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            # Move the data and target tensors to the device
            data = data.to(device)
            target = target.to(device)

            # Forward pass
            output = model(data)

            # Calculate the loss
            loss = loss_fn(output, target)

            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_accuracy += (predicted == target).sum().item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_accuracy / len(dataloader.dataset)

    return avg_loss, accuracy

def training_loop(model, train_dataloader, val_dataloader, optimizer, num_epochs):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_accuracy = train(model, train_dataloader, optimizer)
        val_loss, val_accuracy = evaluate(model, val_dataloader)

        print(f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")

        # Check if the current epoch's accuracy is better than the best accuracy so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            # Save the model checkpoint
            torch.save(model.state_dict(), "best_model.pth")

    print("Training completed!")

value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]

criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

model = PSPNet(
    layers=args.layers,
    classes=args.classes,
    zoom_factor=args.zoom_factor,
    criterion=criterion,
    BatchNorm=torch.nn.BatchNorm2d,
    pretrained=False
)

optimizer = torch.optim.SGD(
    [{'params': model.layer0.parameters()},
        {'params': model.layer1.parameters()},
        {'params': model.layer2.parameters()},
        {'params': model.layer3.parameters()},
        {'params': model.layer4.parameters()},
        {'params': model.ppm.parameters(), 'lr': args.base_lr * 10},
        {'params': model.cls.parameters(), 'lr': args.base_lr * 10},
        {'params': model.aux.parameters(), 'lr': args.base_lr * 10}],
    lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

train_transform = transform.Compose([
    transform.RandScale([args.scale_min, args.scale_max]),
    transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
    transform.RandomGaussianBlur(),
    transform.RandomHorizontalFlip(),
    transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
    transform.ToTensor(),
    transform.Normalize(mean=mean, std=std)])

train_data = dataset.SemData(
    split='train',
    data_root=args.data_root,
    data_list=args.train_list,
    transform=train_transform
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True,
    drop_last=True
)

training_loop(
    model,
    train_loader,
    optimizer,
    criterion,
    args.epochs
)

test_loss, test_accuracy = test(model, test_dataloader)
print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")