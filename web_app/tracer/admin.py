from django.contrib import admin

from .models import Line, Point


class LineAdmin(admin.ModelAdmin):
    list_display = ['pk', 'x0', 'y0', 'x1', 'y1']


class PointAdmin(admin.ModelAdmin):
    list_display = ['pk', 'x0', 'y0']


admin.site.register(Line, LineAdmin)
admin.site.register( Point, PointAdmin)
