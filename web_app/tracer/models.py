from django.db import models


class Line(models.Model):
    length = models.IntegerField()
    angle = models.FloatField()
    transition_absciss = models.IntegerField()
    transition_ordinate = models.IntegerField()


    def __str__(self):
        return f'Линия: длина {self.length}, поворот {self.angle}rad, ' \
               f'смещение {self.transition_absciss, self.transition_ordinate}'


    class Meta:
        verbose_name = 'Линия'
        verbose_name_plural = 'Линии'
        ordering = ['pk']


