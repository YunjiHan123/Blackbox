
from datetime import datetime


class Logger:

    def log(self, message):

        now = datetime.now().strftime("%H:%M:%S")

        print(f"[{now}] {message}")
