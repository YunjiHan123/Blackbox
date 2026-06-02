
from collections import deque
from datetime import datetime


class Logger:

    def __init__(self, history_size=12):

        self.history = deque(maxlen=history_size)

    def log(self, message):

        now = datetime.now().strftime("%H:%M:%S")
        entry = {"time": now, "message": message}
        self.history.appendleft(entry)

        print(f"[{now}] {message}")
        return entry

    def get_recent(self):

        return list(self.history)
