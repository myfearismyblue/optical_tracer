from django.contrib import admin

from .models import Axis, Boundary, BoundaryPoint


class LineAdmin(admin.ModelAdmin):
    list_display = ['pk', 'name', 'side', 'memory_id']


class PointAdmin(admin.ModelAdmin):
    list_display = ['pk', 'x0', 'y0', 'line','id']


class AxisAdmin(admin.ModelAdmin):
    list_display = ['pk','name','direction' ,'x0', 'y0', 'x1', 'y1']


admin.site.register(Boundary, LineAdmin)
admin.site.register(BoundaryPoint, PointAdmin)
admin.site.register(Axis, AxisAdmin)