from django.db import models


class SideView(models.Model):
    side = models.CharField(max_length=10, verbose_name='Сторона')

    def __str__(self):
        return f'{self.side}'

    class Meta:
        verbose_name = 'Сторона'
        verbose_name_plural = 'Стороны'
        ordering = ['pk']


class BoundaryView(models.Model):
    name = models.CharField(max_length=50, verbose_name='Имя')
    side = models.ForeignKey(SideView, on_delete=models.CASCADE)
    memory_id = models.BigIntegerField()
    boundary_serial = models.BinaryField(null=True)

    def __str__(self):
        return f'Граница: {self.name}, сторона: {self.side}'

    class Meta:
        verbose_name = 'Граница'
        verbose_name_plural = 'Границы'
        ordering = ['pk']


class AxisView(models.Model):
    name = models.CharField(max_length=15, default=None)
    x0 = models.IntegerField(default=None)
    y0 = models.IntegerField(default=None)
    x1 = models.IntegerField(default=None)
    y1 = models.IntegerField(default=None)
    direction = models.CharField(max_length=10, default=None)
    memory_id = models.BigIntegerField()

    class Meta:
        verbose_name = 'Ось'
        verbose_name_plural = 'Оси'
        ordering = ['name']


class BeamView(models.Model):
    memory_id = models.IntegerField(default=None)


class VectorView(models.Model):
    x0 = models.IntegerField(default=None)
    y0 = models.IntegerField(default=None)
    beam = models.ForeignKey(BeamView, on_delete=models.CASCADE)


class OpticalSystemView(models.Model):
    name = models.CharField(max_length=50, default=None)
    opt_sys_serial = models.BinaryField(null=True)

    def __str__(self):
        return self.name
