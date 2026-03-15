from django.contrib import admin
from .models import SignalRecord, ScanRecord, CapitalRecord, NewsItem


@admin.register(SignalRecord)
class SignalAdmin(admin.ModelAdmin):
    list_display  = ("symbol","signal","grade","confidence","outcome","profit_pct","created_at")
    list_filter   = ("signal","grade","outcome")
    search_fields = ("symbol",)
    list_editable = ("outcome",)
    ordering      = ("-created_at",)


@admin.register(ScanRecord)
class ScanAdmin(admin.ModelAdmin):
    list_display  = ("scanned_at","pairs_scanned","signals_found","btc_score","btc_trend")
    ordering      = ("-scanned_at",)


@admin.register(CapitalRecord)
class CapitalAdmin(admin.ModelAdmin):
    list_display  = ("date","capital_usd","notes")
    ordering      = ("-date",)


@admin.register(NewsItem)
class NewsAdmin(admin.ModelAdmin):
    list_display  = ("title","source","sentiment","published","currencies")
    list_filter   = ("sentiment",)
    search_fields = ("title","currencies")
    ordering      = ("-published",)