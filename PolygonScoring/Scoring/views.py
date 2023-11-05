from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from django.core.files.storage import FileSystemStorage

import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import shutil
import os

from .forms import UploadFileForm
from .modelscr import save_data, data_import, data_augmentation, model_prediction, ood_detection
from .ada import adv_data_augementation, grad_cam
from django.conf import settings


def index(request):
    shutil.rmtree(settings.MEDIA_ROOT, ignore_errors=True)
    os.mkdir(settings.MEDIA_ROOT)
    return render(request, "html/home.html")


def FileUpload(request):
    if request.method == "POST":
        # form = UploadFileForm(request.POST, request.FILES)
        # if form.is_valid():
        model_file = request.FILES.get('modelfile', False)
        data_zip = request.FILES.get('datafile', False)
        fs = FileSystemStorage()
        file01 = fs.save("model.onnx", model_file)
        file02 = fs.save("dataset.zip", data_zip)
        rs_w = int(request.POST['width'])
        rs_h = int(request.POST['height'])
        aug = {
            'rotation': 0 if 'rt' not in request.POST else 1,
            'flip': 0 if 'flp' not in request.POST else 1,
            'brightness': 0 if 'brt' not in request.POST else 1,
            'contrast': 0 if 'cnt' not in request.POST else 1
        }

        return ScoringResults(request, (rs_w, rs_h), aug)
    else:
        form = UploadFileForm()
    return render(request, "html/FileUpload.html", {"form": form})


def ScoringResults(request, rs_scale, aug_opts):
    save_data()

    X, Y = data_import(rs_scale)
    Y_pred = model_prediction(X)
    Y_pred = np.argmax(Y_pred, axis=1)
    results = {
        'Accuracy': round(accuracy_score(Y, Y_pred), 3),
        'Precision': round(precision_score(Y, Y_pred), 3),
        'Recall': round(recall_score(Y, Y_pred), 3),
        'F1_score': round(f1_score(Y, Y_pred), 3)
    }

    X_aug, Y_aug = data_augmentation(aug_opts)
    Y_pred_aug = model_prediction(X_aug)
    Y_pred_aug = np.argmax(Y_pred_aug, axis=1)
    results_aug = {
        'Accuracy': round(accuracy_score(Y_aug, Y_pred_aug), 3),
        'Precision': round(precision_score(Y_aug, Y_pred_aug), 3),
        'Recall': round(recall_score(Y_aug, Y_pred_aug), 3),
        'F1_score': round(f1_score(Y_aug, Y_pred_aug), 3)
    }

    msp, entropy = ood_detection(X)
    auroc = {
        'AUROC_MSP': round(msp, 3),
        'AUROC_Entropy': round(entropy, 3)
    }

    aug_gc_ind = np.random.choice(np.where(Y == Y_pred)[0], 32)
    gc_ind = np.random.choice(32)
    torch_model, aug_images_t, org_images_t, output_y_n = adv_data_augementation(X[aug_gc_ind], Y[aug_gc_ind])
    gc_ind = np.random.choice(np.where(Y[aug_gc_ind] != output_y_n.detach().numpy())[0])
    grad_cam(torch_model, org_images_t[gc_ind], 'org')
    grad_cam(torch_model, aug_images_t[gc_ind], 'aug')

    return render(request, "html/results.html", {'results': results,
                                                 'results_aug': results_aug,
                                                 'auroc': auroc})
