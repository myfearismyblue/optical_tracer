from django import forms

from .models import SideView


class AddLayer(forms.Form):
    name = forms.CharField(max_length=50)
    side = forms.ModelChoiceField(label='Сторона:', queryset=SideView.objects.all())
    equation = forms.CharField(max_length=50)
