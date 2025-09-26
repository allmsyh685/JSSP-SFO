import sys
from datetime import datetime


class FileOnlyLogger:
    """Write all stdout to a file only (no console mirroring)."""
    def __init__(self, filename: str = "jsp_output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

        self.log.write("Sailfish Job Shop Scheduling Optimizer Output Log\n")
        self.log.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"{'='*80}\n\n")
        self.log.flush()

    def write(self, message: str):
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()

    def close(self):
        if self.log:
            self.log.close()


class StdoutSilencer:
    """Context manager to silence stdout (send to null)."""
    def __init__(self):
        self._original_stdout = sys.stdout
        self._devnull = None

    def __enter__(self):
        import os
        self._devnull = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = self._devnull
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._devnull and not self._devnull.closed:
                self._devnull.flush()
        finally:
            sys.stdout = self._original_stdout
            try:
                if self._devnull and not self._devnull.closed:
                    self._devnull.close()
            except Exception:
                pass
        return False


