"""
dashboard/management/commands/runall.py
Single command: auto-migrates, then starts server + bot together.
Usage: python manage.py runall
"""

import threading
import time
import schedule

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

    def handle(self, *args, **options):
        port = options["port"]
        host = options["host"]

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
            f"  ║  BTC Scalp Bot v{cfg.version}        ║\n"
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
                dash_views.set_scanner(scanner)
                sender  = WhatsAppSender(cfg.whatsapp)
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
        import os, subprocess, sys
        render_port = os.environ.get("PORT")  # Render injects $PORT
        if render_port:
            # On Render: start gunicorn bound to 0.0.0.0:$PORT
            self.stdout.write(f"  Starting gunicorn on 0.0.0.0:{render_port} ...\n\n")
            try:
                subprocess.run([
                    sys.executable, "-m", "gunicorn",
                    "btc_project.wsgi:application",
                    "--bind", f"0.0.0.0:{render_port}",
                    "--workers", "2",
                    "--timeout", "120",
                ], check=True)
            except KeyboardInterrupt:
                pass
        else:
            # Local dev: use Django dev server
            self.stdout.write(f"  Starting server at http://{host}:{port} ...\n\n")
            try:
                call_command("runserver", f"{host}:{port}", use_reloader=False)
            except KeyboardInterrupt:
                self.stdout.write("\n  Shutting down...\n")