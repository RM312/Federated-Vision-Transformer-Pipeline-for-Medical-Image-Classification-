from datetime import datetime

class Logger:

    @staticmethod
    def info(message):
        print(f"[INFO {datetime.now().strftime('%H:%M:%S')}] {message}")

    @staticmethod
    def warning(message):
        print(f"[WARNING] {message}")

    @staticmethod
    def error(message):
        print(f"[ERROR] {message}")