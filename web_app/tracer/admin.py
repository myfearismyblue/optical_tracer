from django.contrib import admin

from .models import Line


class LineAdmin(admin.ModelAdmin):
    list_display = ['pk', 'x0', 'y0', 'x1', 'y1']


admin.site.register(Line, LineAdmin)
