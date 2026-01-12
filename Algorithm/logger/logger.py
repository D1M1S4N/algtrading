import logging
import logging.config
import os
from logging.handlers import RotatingFileHandler

class CustomRotatingFileHandler(RotatingFileHandler):
    def doRollover(self):
        """
        Do a rollover, as described in __init__().
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = f"{self.baseFilename}.{i}.log"
                dfn = f"{self.baseFilename}.{i + 1}.log"
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            dfn = self.baseFilename + ".1.log"
            if os.path.exists(dfn):
                os.remove(dfn)
            self.rotate(self.baseFilename, dfn)
        if not self.delay:
            self.stream = self._open()

# Creamos la carpeta donde guardaremos los logs si no está creada
log_dir = r"logger\logs"
os.makedirs(log_dir, exist_ok=True)

# Guardamos el path a donde guardaremos los logs
log_path = os.path.join(log_dir, "Market_log.log")

# Configuramos el Custom Rotating Handler
file_handler = CustomRotatingFileHandler(
    log_path, 
    maxBytes=1000000, # 1000000 bytes (1 Mb)
    backupCount=3  # mantiene 3 archivos viejos
)
file_handler.setLevel(logging.DEBUG)

# Formato del log
formato = logging.Formatter("[%(levelname)s] [%(asctime)s] [%(module)s] [%(funcName)s Linea: %(lineno)d] - %(message)s")
file_handler.setFormatter(formato)

# Filtro para que no se creen logs identicos seguidos
class DuplicateFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.last_log = None

    def filter(self, record):
        current_log = (record.levelno, record.msg)
        if current_log != self.last_log:
            self.last_log = current_log
            return True
        return False


# Creamos nuestro logger (No usar Logging)
market_logger = logging.getLogger("MarketLogger")
market_logger.setLevel(logging.INFO)
market_logger.addHandler(file_handler)

# Añade el filtro al logger
duplicate_filter = DuplicateFilter()
market_logger.addFilter(duplicate_filter)
