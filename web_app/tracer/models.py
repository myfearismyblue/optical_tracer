from django.db import models


class Line(models.Model):
    length = models.IntegerField(verbose_name='Длина')
    angle = models.FloatField(verbose_name='Угол')
    transition_absciss = models.IntegerField(verbose_name='Абсцисса смещения')
    transition_ordinate = models.IntegerField(verbose_name='Ордината смещения')


    def __str__(self):
        return f'Линия: длина {self.length}, поворот {self.angle}rad, ' \
               f'смещение {self.transition_absciss, self.transition_ordinate}'


    class Meta:
        verbose_name = 'Линия'
        verbose_name_plural = 'Линии'
        ordering = ['pk']


