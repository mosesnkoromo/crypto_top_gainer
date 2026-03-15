from django.urls import path
from . import views

urlpatterns = [
    path("",                                   views.dashboard,           name="dashboard"),
    path("api/stats/",                         views.api_stats,           name="api_stats"),
    path("api/signals/",                       views.api_signals,         name="api_signals"),
    path("api/chart/daily/",                   views.api_chart_daily,     name="api_chart_daily"),
    path("api/grades/",                        views.api_grades,          name="api_grades"),
    path("api/pairs/",                         views.api_top_pairs,       name="api_pairs"),
    path("api/scans/",                         views.api_scans,           name="api_scans"),
    path("api/outcome-distribution/",          views.api_outcome_distribution, name="api_outcome_dist"),
    path("api/capital/",                       views.api_capital,         name="api_capital"),
    path("api/capital/add/",                   views.api_capital_add,     name="api_capital_add"),
    path("api/report/",                        views.api_report,          name="api_report"),
    path("api/signal/<int:signal_id>/outcome/",views.api_update_outcome,  name="api_outcome"),
]
