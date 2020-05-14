# Модель - Unet (heatmap regression)
MODEL_NAME = 'unet-heatmap-regression'

# Импорты
import os
import sys
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from dataset import ThousandLandmarksDataset
from augmentations import (
    ScaleMinSideToSize,
    CropCenter,
    TransformByKeys,
    Heatmap
)

from routines import train, validate, predict, create_submission, train_iter
from models.unet import UNet
from losses import WingLoss, AdaptiveWingLoss


# Повторяющийся рандом
np.random.seed(1234)
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Справочная информация

## Расположение точек

# - 0: 273 - овал лица
# - 273: 337 - левая бровь
# - 337: 401 - правая бровь
# - 401: 527 - нос
# - 527: 587 - вертикальная линия от губ до лба по носу
# - 587: 714 - левый глаз(контур и яблоко)
# - 714: 841 - правый глаз(контур и яблоко)
# - 841: 906 - верхняя губа
# - 906: 969 - нижняя губа
# - 969: 970 - центр левого глаза
# - 970: 971 - центр правого глаза
# - 650: 651 - точка левого глаза слева
# - 682: 683 - точка левого глаза справа
# - 777: 778 - точка правого глаза справа
# - 808: 809 - точка правого глаза слева

## Константы

# Отрезаемый размер
CROP_SIZE = 128

# Число точек для предсказания
NUM_PTS = 971

# Процент тренировочной выборки при разбиении
TRAIN_SIZE = 0.8

# Размер батча
TRAIN_BATCH_SIZE = 16

# Чтение данных
TRAIN_DATA_PATH = '/home/kovalexal/Spaces/learning/made/made_cv/competitions/facial_points/data/train/'

train_transforms = transforms.Compose([
    # Базовая предобработка
    ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
    CropCenter(CROP_SIZE),

    # Переводим ключевые точки в heatmap
    Heatmap((3, 3)),

    # Обработка для подачи на обучение
    TransformByKeys(transforms.ToPILImage(), ('image',)),
    TransformByKeys(transforms.ToTensor(), ('image',)),
    TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ('image',)),
])

val_transforms = transforms.Compose([
    # Базовая предобработка
    ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
    CropCenter(CROP_SIZE),

    # Переводим ключевые точки в heatmap
    Heatmap((3, 3)),

    # Обработка для подачи на обучение
    TransformByKeys(transforms.ToPILImage(), ('image',)),
    TransformByKeys(transforms.ToTensor(), ('image',)),
    TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ('image',)),
])

train_dataset = ThousandLandmarksDataset(TRAIN_DATA_PATH, train_transforms, split='train', TRAIN_SIZE=TRAIN_SIZE)
print(len(train_dataset))
val_dataset = ThousandLandmarksDataset(TRAIN_DATA_PATH, val_transforms, split='val', TRAIN_SIZE=TRAIN_SIZE)
print(len(val_dataset))

# Обучение и валидация
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=0, pin_memory=True, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=0, pin_memory=True, shuffle=False, drop_last=False)

# learning-rate
LEARNING_RATE = 1e-3

# Число эпох
N_EPOCHS = 10

# tensorboard
writer = SummaryWriter(log_dir='./{}'.format(MODEL_NAME), comment=MODEL_NAME)

# Задаем модель
model = UNet(3, NUM_PTS)
model.to(device)
with torch.no_grad():
    # writer.add_graph(model, next(iter(val_dataloader))['image'].to(device))
    summary(model, next(iter(train_dataloader))['image'].shape[1:])

# Задаем параметры оптимизации
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)
# criterion = F.mse_loss
criterion = AdaptiveWingLoss()

# Временные параметры для выбора наилучшего результата
best_val_loss, best_model_state_dict = np.inf, {}

CURRENT_EPOCH = 0

for epoch in range(CURRENT_EPOCH, N_EPOCHS):
    train_loss = train_iter(MODEL_NAME, 470, epoch, model, train_dataloader, criterion, optimizer, device=device, writer=writer, log_every=10)
    writer.add_scalar('EpochLoss/train', train_loss, epoch)
    with open('{}_train_{}.pth'.format(MODEL_NAME, epoch), 'wb') as fp:
        torch.save(model.state_dict(), fp)

    val_loss = validate(epoch, model, val_dataloader, criterion, device=device, writer=writer, log_every=20)
    writer.add_scalar('EpochLoss/val', val_loss, epoch)

    print('Epoch #{:2}:\ttrain loss: {:5.5}\tval loss: {:5.5}'.format(epoch, train_loss, val_loss))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state_dict = model.state_dict()
        with open('{}_best.pth'.format(MODEL_NAME), 'wb') as fp:
            torch.save(model.state_dict(), fp)

    CURRENT_EPOCH += 1


# Предсказание и сохранение результата
TEST_DATA_PATH = '/home/kovalexal/Spaces/learning/made/made_cv/competitions/facial_points/data/test/'
test_dataset = ThousandLandmarksDataset(TEST_DATA_PATH, train_transforms, split='test')

# Размер батча
TEST_BATCH_SIZE = 2
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=False, drop_last=False)

with open('{}_best.pth'.format(MODEL_NAME), 'rb') as fp:
    best_state_dict = torch.load(fp, map_location="cpu")
    model.load_state_dict(best_state_dict)

test_predictions = predict(model, test_dataloader, device)
with open('{}_test_predictions.pkl'.format(MODEL_NAME), 'wb') as fp:
    pickle.dump({'image_names': test_dataset.image_names, 'landmarks': test_predictions}, fp)

create_submission(TEST_DATA_PATH, test_predictions, '{}_submit.csv'.format(MODEL_NAME))
