from django.contrib import admin

from .models import Line, Point


class LineAdmin(admin.ModelAdmin):
    list_display = ['pk', 'name', 'side', 'memory_id']


class PointAdmin(admin.ModelAdmin):
    list_display = ['pk', 'x0', 'y0', 'id']


admin.site.register(Line, LineAdmin)
admin.site.register( Point, PointAdmin)
