from django.contrib import admin

from .models import Line


class LineAdmin(admin.ModelAdmin):
    list_display = ['pk', 'length', 'angle']


admin.site.register(Line, LineAdmin)
