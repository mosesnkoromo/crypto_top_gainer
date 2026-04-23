"""
dashboard/management/commands/runall.py
Single command: auto-migrate, then start server + bot together.
Usage: python manage.py runall          # uses env WORKERS or default 2
       python manage.py runall --workers=3
"""

import threading
import time
import schedule
import os
import sys
import subprocess

from django.core.management.base import BaseCommand
from django.core.management import call_command

from config import load_config
from src.alerts.whatsapp import WhatsAppSender
from src.scanner import Scanner
from src.utils.formatter import fmt_startup
from src.utils.logger import get_logger, setup_logging


class Command(BaseCommand):
    help = "Auto-migrate, then start Django server + bot in one command"

    def add_arguments(self, parser):
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--host", type=str, default="127.0.0.1")
        parser.add_argument("--workers", type=int,
                            default=int(os.environ.get("WORKERS", 2)),
                            help="Number of gunicorn workers (only used in production)")

    def handle(self, *args, **options):
        port = options["port"]
        host = options["host"]
        workers = options["workers"]

        # ── Step 1: Auto-migrate ──────────────────────────────
        self.stdout.write("  Running migrations...")
        try:
            call_command("makemigrations", "dashboard", verbosity=0)
            call_command("migrate", verbosity=0)
            self.stdout.write(self.style.SUCCESS("  ✅ Database ready"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  ❌ Migration failed: {e}"))
            return

        # ── Step 2: Load config + logging ────────────────────
        cfg = load_config()
        setup_logging(
            cfg.log.log_dir, cfg.log.log_filename,
            cfg.log.level, cfg.log.max_bytes, cfg.log.backup_count,
        )
        log = get_logger(__name__)

        self.stdout.write(self.style.SUCCESS(
            f"\n"
            f"  ╔══════════════════════════════════╗\n"
            f"  ║  BTC Strength Bot v{cfg.version}        ║\n"
            f"  ╚══════════════════════════════════╝\n"
            f"\n"
            f"  Dashboard : http://{host}:{port}\n"
            f"  Scanner   : every {cfg.scan.scan_interval_minutes} min\n"
            f"  Logs      : {cfg.log.log_dir / cfg.log.log_filename}\n"
        ))

        # ── Step 3: Start bot in background thread ────────────
        def run_bot():
            log.info("Bot thread starting...")
            try:
                scanner = Scanner(cfg)
                from dashboard import views as dash_views
                dash_views.api_scans(scanner)
                sender = WhatsAppSender(cfg.whatsapp)
                sender.send(fmt_startup(
                    cfg.version, cfg.scan.scan_interval_minutes,
                    cfg.scan.top_gainers_count, cfg.scan.min_gain_percent,
                    cfg.risk.tp1_pct, cfg.risk.tp2_pct, cfg.risk.tp3_pct, cfg.risk.sl_pct,
                ))
                scanner.run_cycle()
                schedule.every(cfg.scan.scan_interval_minutes).minutes.do(scanner.run_cycle)
                log.info("Bot scheduler active — every %d min", cfg.scan.scan_interval_minutes)
                while True:
                    schedule.run_pending()
                    time.sleep(20)
            except Exception as e:
                log.error("Bot thread crashed: %s", e, exc_info=True)

        bot_thread = threading.Thread(target=run_bot, daemon=True, name="BotScanner")
        bot_thread.start()

        # ── Step 4: Start server ──────────────────────────────
        render_port = os.environ.get("PORT")
        if render_port:
            # Production: gunicorn
            self.stdout.write(
                f"  Starting gunicorn on 0.0.0.0:{render_port} with {workers} workers...\n\n"
            )
            subprocess.run([
                sys.executable, "-m", "gunicorn",
                "btc_project.wsgi:application",
                "--bind", f"0.0.0.0:{render_port}",
                "--workers", str(workers),
                "--timeout", "120",
                "--access-logfile", "-",
            ])
        else:
            # Local dev
            self.stdout.write(f"  Starting dev server at http://{host}:{port} ...\n\n")
            call_command("runserver", f"{host}:{port}", use_reloader=False)