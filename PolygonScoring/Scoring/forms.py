from django import forms


class UploadFileForm(forms.Form):
    model_file = forms.FileField()
    data_zip = forms.FileField()