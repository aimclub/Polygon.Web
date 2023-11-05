import PIL.Image
import numpy as np
import pandas as pd
import os
import zipfile
from PIL import Image, ImageEnhance
import torch
import torchvision.transforms as TF
from sklearn.metrics import roc_auc_score, roc_curve

from django.conf import settings

import onnx
import onnxruntime


def model_import(model_path):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    return None


def save_data():
    path_to_zip_file = settings.MEDIA_ROOT + '/dataset.zip'
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(settings.MEDIA_ROOT)

    os.rename(settings.MEDIA_ROOT + "/lung_CT",
              settings.MEDIA_ROOT + "/dataset")


def image_prepr(img, rs_scale):
    _normalize = TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tfm = TF.Compose([TF.Resize((rs_scale[0], rs_scale[1])), TF.ToTensor(), _normalize])

    img = tfm(img)

    return np.asarray(img)


def data_import(rs_scale):
    markup = pd.read_csv(settings.MEDIA_ROOT + '/dataset/markup.csv', sep=',')

    filenames_dir = settings.MEDIA_ROOT + "/dataset/data"
    filenames = markup.filename.values
    # filenames = os.listdir(filenames_dir)
    # Images import
    # data = np.array([np.load(filenames_dir + '/' + fname) for fname in filenames])
    data = []

    for fname in filenames:
        img = Image.open(filenames_dir + '/' + fname).convert('RGB')
        img = image_prepr(img, rs_scale)
        data.append(img)

    data = np.array(data)

    target = []
    for obj in filenames:
        target.append(markup[markup.filename == obj].label.values[0])
    target = np.array(target)

    return data, target


def data_augmentation(augmet_opts):
    markup = pd.read_csv(settings.MEDIA_ROOT + '/dataset/markup.csv', sep=',')

    filenames_dir = settings.MEDIA_ROOT + "/dataset/data"
    # filenames = os.listdir(filenames_dir)
    filenames = markup.filename.values

    files_aug = np.random.choice(filenames, size=len(filenames) // 2, replace=False)
    aug_mask = np.isin(filenames, files_aug, invert=True)
    files_org = filenames[aug_mask]

    data = []
    for fname in files_aug:
        img = Image.open(filenames_dir + '/' + fname).convert('RGB')
        if augmet_opts['rotation']:
            img = img.rotate(np.random.randint(10, 46), expand=True)
        if augmet_opts['flip']:
            flp_v = np.random.randint(1, 3)
            if flp_v == 1:
                img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            if flp_v == 2:
                img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        if augmet_opts['brightness']:
            factor = np.random.uniform(0.5, 2)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        if augmet_opts['contrast']:
            factor = np.random.uniform(0.5, 2)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)

        img = image_prepr(img, (256, 256))
        data.append(img)

    for fname in files_org:
        img = Image.open(filenames_dir + '/' + fname).convert('RGB')
        img = image_prepr(img, (256, 256))
        data.append(img)

    data = np.array(data)

    target = []
    for obj in files_aug:
        target.append(markup[markup.filename == obj].label.values[0])
    for obj in files_org:
        target.append(markup[markup.filename == obj].label.values[0])
    target = np.array(target)

    return data, target


def ood_detection(in_data):
    out_data = np.load(str(settings.BASE_DIR) + "/brain_ct.npy")
    out_data = out_data[:in_data.shape[0]]

    ood_data = np.concatenate((in_data, out_data), axis=0)
    ood_labels = np.array([1] * len(in_data) + [0] * len(out_data))

    preds = model_prediction(ood_data)
    probs = np.max(preds, axis=1)

    entr = np.sum(np.log(abs(preds[:])) * abs(preds[:]), axis=1)

    msp = roc_auc_score(ood_labels, probs)
    entropy = roc_auc_score(ood_labels, entr)

    return msp, entropy


def model_prediction(data):
    ort_session = onnxruntime.InferenceSession(settings.MEDIA_ROOT + "/model.onnx")

    ort_inputs = {ort_session.get_inputs()[0].name: data}
    ort_outs = ort_session.run(None, ort_inputs)
    # ort_outs = np.argmax(ort_outs[0], axis=1)

    return ort_outs[0]
