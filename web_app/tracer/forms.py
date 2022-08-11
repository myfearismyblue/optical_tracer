from django import forms


class AddLayer(forms.Form):
    name = forms.CharField(max_length=50)
    side = forms.CharField(max_length=10)
    equation = forms.CharField(max_length=50)
