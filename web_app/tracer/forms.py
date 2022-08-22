from django import forms

from .models import SideView, OpticalSystemView


class AddComponent(forms.Form):
    first_layer_name = forms.CharField(max_length=50, label='Имя первой границы')
    first_layer_side = forms.ModelChoiceField(label='Сторона:',
                                              queryset=SideView.objects.all(),
                                              empty_label=None)
    first_layer_equation = forms.CharField(max_length=50, label='Уравнение поверхности')
    second_layer_name = forms.CharField(max_length=50, label='Имя первой границы')
    second_layer_side = forms.ModelChoiceField(label='Сторона:',
                                               queryset=SideView.objects.all(),
                                               empty_label=None)
    second_layer_equation = forms.CharField(max_length=50, label='Уравнение поверхности')
    material_name = forms.CharField(max_length=50, label='Название среды')
    # light absorption in % while tracing 1 sm of thickness
    transmittance = forms.FloatField(max_value=100, min_value=0, label='Ослабление среды')
    index = forms.FloatField(max_value=3, min_value=1, label='Показатель преломления среды')
    component_name = forms.CharField(max_length=50, label='Название компонента')


class ChooseOpticalSystem(forms.Form):
    optical_system = forms.ModelChoiceField(label='Оптическая система:',
                                            queryset=OpticalSystemView.objects.all(),
                                            empty_label='Новая оптическая система')
