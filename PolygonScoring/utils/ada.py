from onnx2torch import convert
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as TF
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.transforms.functional import to_pil_image

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps
import PIL
from PIL import Image

from django.conf import settings

matplotlib.use('agg')


def loss_fn(y_true, y_pred, lhl_1, lhl_2):
    sftmx = nn.Softmax(dim=1)
    cross_entropy_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    sm_crossentr = cross_entropy_loss(sftmx(y_pred), y_true)
    mse = mse_loss(lhl_1, lhl_2)
    loss = sm_crossentr - 1.0 * mse
    return loss


def adv_data_augementation(data, labels):
    onnx_model_path = settings.MEDIA_ROOT + "/model.onnx"
    torch_model = convert(onnx_model_path)

    for param in torch_model.parameters():
        param.requires_grad = False

    lhl_name = get_graph_node_names(torch_model)[0][-2]
    out_name = get_graph_node_names(torch_model)[0][-1]

    lhl_model_out = create_feature_extractor(torch_model, return_nodes={lhl_name: 'lhl', out_name: 'out'})

    aug_labels_t = torch.tensor(labels)
    aug_images_t = torch.tensor(data, requires_grad=True)
    optimizer = optim.SGD([aug_images_t], lr=1.0, maximize=True)

    aug_images_org = aug_images_t.clone()
    output_lhl = lhl_model_out(aug_images_org)['lhl']
    for i in range(16):
        aug_images_org = aug_images_t.clone()
        output_n = lhl_model_out(aug_images_org)
        output_y_n = output_n['out']
        output_lhl_n = output_n['lhl']

        loss = loss_fn(aug_labels_t, output_y_n, output_lhl, output_lhl_n)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    result_aug = aug_images_t.clone()
    return torch_model, result_aug, torch.tensor(data), output_y_n.argmax(dim=1)


def backward_hook(module, grad_input, grad_output):
    global gc_gradients
    gc_gradients = grad_output


def forward_hook(module, args, output):
    global gc_activations
    gc_activations = output


gc_gradients = None
gc_activations = None


def grad_cam(torch_model, image, name):
    last_conv_name = [l for l in get_graph_node_names(torch_model)[0] if 'Conv' in l][-1]
    torch_model.get_parameter(last_conv_name + '.weight').requires_grad = True
    torch_model.get_parameter(last_conv_name + '.bias').requires_grad = True

    back_hook = torch_model.get_submodule(last_conv_name).register_full_backward_hook(backward_hook, prepend=False)
    for_hook = torch_model.get_submodule(last_conv_name).register_forward_hook(forward_hook, prepend=False)

    gc_out = torch_model(image.unsqueeze(0))
    torch_model(image.unsqueeze(0)).backward(gc_out)

    pooled_gradients = torch.mean(gc_gradients[0], dim=[0, 2, 3])
    for i in range(gc_activations.size()[1]):
        gc_activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(gc_activations, dim=1).squeeze()
    heatmap = nn.functional.relu(heatmap)
    heatmap /= torch.max(heatmap)

    de_normalize = TF.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    to_pil_image(de_normalize(image), mode='RGB').save(settings.MEDIA_ROOT + '/' + name + '_image.png')

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(to_pil_image(de_normalize(image), mode='RGB'))

    overlay = to_pil_image(heatmap.detach(), mode='F').resize((256, 256), resample=PIL.Image.BICUBIC)
    cmap = colormaps['jet']
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    ax.imshow(overlay, alpha=0.4, interpolation='nearest')

    plt.savefig(settings.MEDIA_ROOT + '/' + name + '_gradcam.png', bbox_inches='tight')

    back_hook.remove()
    for_hook.remove()
