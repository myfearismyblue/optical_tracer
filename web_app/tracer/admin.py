from django.contrib import admin

from .models import AxisView, BoundaryView


class LineAdmin(admin.ModelAdmin):
    list_display = ['pk', 'name', 'side', 'memory_id']


class AxisAdmin(admin.ModelAdmin):
    list_display = ['pk','name','direction' ,'x0', 'y0', 'x1', 'y1']


admin.site.register(BoundaryView, LineAdmin)
admin.site.register(AxisView, AxisAdmin)