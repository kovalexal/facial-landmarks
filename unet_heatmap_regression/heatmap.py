import math

import torch


def landmarks2heatmap(height, width, centers, sigma=(0.2, 0.2)):
    """Осуществляет преобразование landmark'ов в heatmap для одного изображения

    :param height: Высота изображения
    :param width: Ширина изображения
    :param centers: Центры ключевых точек
    :param sigma: Сигма для нормального распределения
    :return: (n_points, height, width)
    """

    # scalex = 1 / (sigma[0] * math.sqrt(2 * math.pi))
    # scaley = 1 / (sigma[1] * math.sqrt(2 * math.pi))

    x = centers[:, 0].reshape(-1, 1) - torch.arange(height).float()
    # x = scalex * torch.exp(-0.5 * torch.pow(x / sigma[0], 2))
    x = torch.exp(-0.5 * torch.pow(x / sigma[0], 2))
    x = x.contiguous().view(-1, height, 1)

    y = centers[:, 1].reshape(-1, 1) - torch.arange(width).float()
    # y = scaley * torch.exp(-0.5 * torch.pow(y / sigma[1], 2))
    y = torch.exp(-0.5 * torch.pow(y / sigma[1], 2))
    y = y.contiguous().view(-1, width, 1)

    return torch.matmul(x, y.transpose(1, 2))#.permute(1, 2, 0)


# def heatmap2landmarks(heatmaps):
#     """Преобразует heatmap в набор ключевых точек для батча
#
#     :param heatmaps: heatmap'ы для батча (batch_size, n_batch)
#     :return:
#     """
#     b, _, _, _ = heatmaps.shape
#     xs = heatmaps.argmax(2)[:, :, 0].reshape(-1)
#     ys = heatmaps.argmax(3)[:, :, 0].reshape(-1)
#     landmarks = torch.stack([xs, ys], dim=1)
#     return landmarks.reshape(b, xs.shape[0] // b, -1).reshape(b, -1)


def heatmaps2landmarks(heatmaps):
    """Восстанавливает ключевые точки по heatmap'ам c помощью argmax

    :param heatmaps: Набор ключевых точек (batch_size, n_points, d, d)
    :return: (batch_size, n_points, 2)
    """
    b, n, d, _ = heatmaps.shape
    m = heatmaps.view(b, n, -1).argmax(2)
    x, y = m // d, m % d
    return torch.cat((x[:, :, None], y[:, :, None]), dim=2)


def heatmaps2landmarks2(heatmaps):
    """Восстанавливает ключевые точки по heatmap'ам усреднением по распределению

    :param heatmaps: Набор ключевых точек (batch_size, n_points, d, d)
    :return: (batch_size, n_points, 2)
    """

    b, n, d, _ = heatmaps.shape

    # Нормализуем гистограммы
    heatmaps_normalized = torch.div(heatmaps, heatmaps.sum(dim=(2, 3))[:, :, None, None])

    # Получаем координаты
    x = (torch.arange(d)[None, None, :, None] * heatmaps_normalized).sum(dim=(2, 3))
    y = (torch.arange(d)[None, None, None, :] * heatmaps_normalized).sum(dim=(2, 3))

    return torch.cat((x[:, :, None], y[:, :, None]), dim=2)
