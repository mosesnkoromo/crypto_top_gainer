"""
dashboard/management/commands/runbot.py
Run the trading bot as a Django management command:
  python manage.py runbot
"""

import schedule
import time
from django.core.management.base import BaseCommand
from config import load_config
from src.alerts.whatsapp import WhatsAppSender
from src.scanner import Scanner
from src.utils.formatter import fmt_startup
from src.utils.logger import get_logger, setup_logging


class Command(BaseCommand):
    help = "Run the BTC Strength WhatsApp alert bot"

    def handle(self, *args, **options):
        cfg = load_config()
        setup_logging(cfg.log.log_dir, cfg.log.log_filename,
                      cfg.log.level, cfg.log.max_bytes, cfg.log.backup_count)
        log = get_logger(__name__)
        log.info("Bot v%s starting via Django management command", cfg.version)

        scanner = Scanner(cfg)
        sender  = WhatsAppSender(cfg.whatsapp)
        sender.send(fmt_startup(
            cfg.version, cfg.scan.scan_interval_minutes,
            cfg.scan.top_gainers_count, cfg.scan.min_gain_percent,
            cfg.risk.tp1_pct, cfg.risk.tp2_pct, cfg.risk.tp3_pct, cfg.risk.sl_pct,
        ))

        scanner.run_cycle()
        schedule.every(cfg.scan.scan_interval_minutes).minutes.do(scanner.run_cycle)

        self.stdout.write(self.style.SUCCESS(
            f"Bot running — scanning every {cfg.scan.scan_interval_minutes} min"
        ))

        while True:
            schedule.run_pending()
            time.sleep(20)
