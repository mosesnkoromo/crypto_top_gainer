web: gunicorn btc_project.wsgi:application --bind 0.0.0.0:$PORT --workers 2 --timeout 120
worker: python manage.py runbot

