import os
import sys
# Импортируем tqdm
try:
    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
        # Jupyter notebook
        from tqdm.notebook import tqdm_notebook as tqdm
    else:
        raise NameError
except NameError:
    from tqdm import tqdm

# Остальные import'ы
import torch
import numpy as np
import pandas as pd
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SUBMISSION_HEADER = "file_name,Point_M0_X,Point_M0_Y,Point_M1_X,Point_M1_Y,Point_M2_X,Point_M2_Y,Point_M3_X,Point_M3_Y,Point_M4_X,Point_M4_Y,Point_M5_X,Point_M5_Y,Point_M6_X,Point_M6_Y,Point_M7_X,Point_M7_Y,Point_M8_X,Point_M8_Y,Point_M9_X,Point_M9_Y,Point_M10_X,Point_M10_Y,Point_M11_X,Point_M11_Y,Point_M12_X,Point_M12_Y,Point_M13_X,Point_M13_Y,Point_M14_X,Point_M14_Y,Point_M15_X,Point_M15_Y,Point_M16_X,Point_M16_Y,Point_M17_X,Point_M17_Y,Point_M18_X,Point_M18_Y,Point_M19_X,Point_M19_Y,Point_M20_X,Point_M20_Y,Point_M21_X,Point_M21_Y,Point_M22_X,Point_M22_Y,Point_M23_X,Point_M23_Y,Point_M24_X,Point_M24_Y,Point_M25_X,Point_M25_Y,Point_M26_X,Point_M26_Y,Point_M27_X,Point_M27_Y,Point_M28_X,Point_M28_Y,Point_M29_X,Point_M29_Y\n"

####################################################################################
##################################### Обучение #####################################
####################################################################################

def train(epoch, model, iterator, criterion, optimizer, device=DEVICE, log_every=100, writer=None):
    """Рутина обучения

    :param epoch: Номер текущей эпохи (необходимо для логирования)
    :param model: Сеть
    :param iterator: Итератор (инстанс Dataloader)
    :param criterion: Loss функция
    :param optimizer: Оптимизатор
    :param device: Устройство для осуществления расчетов
    :param log_every: Раз в сколько итераций осуществлять логирование в writer
    :param writer: writer для tensorboard
    :return: Средний loss на этой эпохе
    """

    model.train()

    train_loss = []
    for i, batch in enumerate(tqdm(iterator, total=len(iterator), desc='training...')):
        images = batch['image'].to(device)
        landmarks = batch['landmarks']
        landmarks.requires_grad_(False)

        pred_landmarks = model(images).cpu()
        pred_landmarks.requires_grad_(True)
        loss = criterion(pred_landmarks, landmarks)
        train_loss.append(loss.item())

        if writer is not None:
            if ((i + 1) % log_every) == 0:
                global_step = epoch * (len(iterator) // log_every) + (i + 1) // log_every
                writer.add_scalar('BatchLoss/train', train_loss[-1], global_step=global_step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(train_loss)


def train_iter(model_name, n, epoch, model, iterator, criterion, optimizer, device=DEVICE, log_every=100, writer=None):
    """Рутина обучения

    :param model_name: Имя модели (необходимо для логирования)
    :param n: Номер текущего батча внутри эпохи
    :param epoch: Номер текущей эпохи (необходимо для логирования)
    :param model: Сеть
    :param iterator: Итератор (инстанс Dataloader)
    :param criterion: Loss функция
    :param optimizer: Оптимизатор
    :param device: Устройство для осуществления расчетов
    :param log_every: Раз в сколько итераций осуществлять логирование в writer
    :param writer: writer для tensorboard
    :return: Средний loss на этой эпохе
    """

    torch.cuda.empty_cache()
    model.train()

    train_loss = []
    for i, batch in enumerate(tqdm(iterator, total=len(iterator), desc='training...')):
        # Пропускаем уже обработанные куски
        if i < n:
            if ((i + 1) % log_every) == 0:
                with open('{}_train_{}_{}.pth'.format(model_name, epoch, i + 1), 'rb') as fp:
                    print('loading: {}_train_{}_{}.pth'.format(model_name, epoch, i + 1))
                    model.load_state_dict(torch.load(fp, map_location="cpu"))
            continue

        images = batch['image'].to(device)
        landmarks = batch['landmarks']

        pred_landmarks = model(images).cpu()
        loss = criterion(pred_landmarks, landmarks)
        train_loss.append(loss.item())

        if writer is not None:
            if ((i + 1) % log_every) == 0:
                global_step = epoch * (len(iterator) // log_every) + (i + 1) // log_every
                writer.add_scalar('BatchLoss/train', train_loss[-1], global_step=global_step)

                with open('{}_train_{}_{}.pth'.format(model_name, epoch, i + 1), 'wb') as fp:
                    torch.save(model.state_dict(), fp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(train_loss)

####################################################################################
#################################### Валидация #####################################
####################################################################################

def validate(epoch, model, iterator, criterion, device=DEVICE, log_every=10, writer=None):
    """Рутина валидации

    :param epoch: Номер текущей эпохи (необходимо для логирования)
    :param model: Сеть
    :param iterator: Итератор (инстанс Dataloader)
    :param criterion: Loss функция
    :param device: Устройство для осуществления расчетов
    :param log_every: Раз в сколько итераций осуществлять логирование в writer
    :param writer: writer для tensorboard
    :return: Средний loss на валидации
    """
    model.eval()

    val_loss = []
    for i, batch in enumerate(tqdm(iterator, total=len(iterator), desc='validation...')):
        images = batch['image'].to(device)
        landmarks = batch['landmarks']

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
            loss = criterion(pred_landmarks, landmarks, reduction='mean')
            val_loss.append(loss.item())

            if writer is not None:
                if ((i + 1) % log_every) == 0:
                    global_step = epoch * (len(iterator) // log_every) + (i + 1) // log_every
                    writer.add_scalar('BatchLoss/val', val_loss[-1], global_step=global_step)

    return np.mean(val_loss)


def validate_iter(model_name, n, epoch, model, iterator, criterion, device=DEVICE, log_every=10, writer=None):
    """Рутина валидации

    :param model_name: Имя модели (необходимо для логирования)
    :param n: Номер текущего батча внутри эпохи
    :param epoch: Номер текущей эпохи (необходимо для логирования)
    :param model: Сеть
    :param iterator: Итератор (инстанс Dataloader)
    :param criterion: Loss функция
    :param device: Устройство для осуществления расчетов
    :param log_every: Раз в сколько итераций осуществлять логирование в writer
    :param writer: writer для tensorboard
    :return: Средний loss на валидации
    """
    torch.cuda.empty_cache()
    model.eval()

    val_loss = []
    for i, batch in enumerate(tqdm(iterator, total=len(iterator), desc='validation...')):
        # Пропускаем уже обработанные куски
        if i < n:
            continue

        images = batch['image'].to(device)
        landmarks = batch['landmarks']

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
            loss = criterion(pred_landmarks, landmarks, reduction='mean')
            val_loss.append(loss.item())

            if writer is not None:
                if ((i + 1) % log_every) == 0:
                    global_step = epoch * (len(iterator) // log_every) + (i + 1) // log_every
                    writer.add_scalar('BatchLoss/val', val_loss[-1], global_step=global_step)

                    # with open('{}_val_{}_{}.pth'.format(model_name, epoch, i + 1), 'wb') as fp:
                    #     torch.save(model.state_dict(), fp)

    return np.mean(val_loss)

####################################################################################
#################################### Применение ####################################
####################################################################################

def restore_landmarks(landmarks, f, margins):
    dx, dy = margins
    landmarks[:, 0] += dx
    landmarks[:, 1] += dy
    landmarks /= f
    return landmarks


def restore_landmarks_batch(landmarks, fs, margins_x, margins_y):
    landmarks[:, :, 0] += margins_x[:, None]
    landmarks[:, :, 1] += margins_y[:, None]
    landmarks /= fs[:, None, None]
    return landmarks


def predict(model, iterator, device):
    """Рутина предсказания

    :param model: Модель
    :param iterator: Итератор (инстанс Dataloader)
    :param device: Устройство для осуществления расчетов
    :return: Предсказания модели
    """
    model.eval()

    predictions = np.zeros((len(iterator.dataset), 971, 2))
    for i, batch in enumerate(tqdm(iterator, total=len(iterator), desc="test prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), 971, 2))  # B x NUM_PTS x 2

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        predictions[i * iterator.batch_size: (i + 1) * iterator.batch_size] = prediction

    return predictions


def create_submission(path_to_data, test_predictions, path_to_submission_file):
    output_file = path_to_submission_file
    wf = open(output_file, 'w')
    wf.write(SUBMISSION_HEADER)

    mapping_path = os.path.join(path_to_data, 'test_points.csv')
    mapping = pd.read_csv(mapping_path, delimiter='\t')

    for i, row in mapping.iterrows():
        file_name = row[0]
        point_index_list = np.array(eval(row[1]))
        points_for_image = test_predictions[i]
        needed_points = points_for_image[point_index_list].astype(np.int)
        wf.write(file_name + ',' + ','.join(map(str, needed_points.reshape(2 * len(point_index_list)))) + '\n')



###################################################################################
####################################### UNet ######################################
###################################################################################

from heatmap import heatmaps2landmarks, heatmaps2landmarks2


def predict_unet(model, iterator, device):
    """Рутина предсказания

    :param model: Модель
    :param iterator: Итератор (инстанс Dataloader)
    :param device: Устройство для осуществления расчетов
    :return: Предсказания модели
    """
    model.eval()

    predictions = np.zeros((len(iterator.dataset), 971, 2))
    for i, batch in enumerate(tqdm(iterator, total=len(iterator), desc="test prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = heatmaps2landmarks(model(images).cpu()).float()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), 971, 2))  # B x NUM_PTS x 2

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        predictions[i * iterator.batch_size: (i + 1) * iterator.batch_size] = prediction

    return predictions


def predict_unet2(model, iterator, device):
    """Рутина предсказания

    :param model: Модель
    :param iterator: Итератор (инстанс Dataloader)
    :param device: Устройство для осуществления расчетов
    :return: Предсказания модели
    """
    model.eval()

    predictions = np.zeros((len(iterator.dataset), 971, 2))
    for i, batch in enumerate(tqdm(iterator, total=len(iterator), desc="test prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = heatmaps2landmarks2(model(images).cpu()).float()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), 971, 2))  # B x NUM_PTS x 2

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        predictions[i * iterator.batch_size: (i + 1) * iterator.batch_size] = prediction

    return predictions


# def create_submission_unet(path_to_data, test_predictions, path_to_submission_file):
#     output_file = path_to_submission_file
#     wf = open(output_file, 'w')
#     wf.write(SUBMISSION_HEADER)
#
#     mapping_path = os.path.join(path_to_data, 'test_points.csv')
#     mapping = pd.read_csv(mapping_path, delimiter='\t')
#
#     for i, row in mapping.iterrows():
#         file_name = row[0]
#         point_index_list = np.array(eval(row[1]))
#         points_for_image = test_predictions[i]
#         needed_points = points_for_image[point_index_list].astype(np.int)
#         wf.write(file_name + ',' + ','.join(map(str, needed_points.reshape(2 * len(point_index_list)))) + '\n')
