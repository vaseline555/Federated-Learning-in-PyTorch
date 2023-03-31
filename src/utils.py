import io
import os
import sys
import torch
import random
import logging

import numpy as np
from multiprocessing import Process

logger = logging.getLogger(__name__)



#########################
# Argparser Restriction #
#########################
class Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
    def __eq__(self, other):
        return self.start <= other <= self.end

########
# Seed #
########
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f'[SEED] ...seed is set ({seed})!')
    
##################
# Logging Add-on #
##################
class TqdmToLogger(io.StringIO):
    def __init__(self, logger):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = logging.INFO
        
    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')
        
    def flush(self):
        self.logger.log(level=self.level, msg=self.buf)   
        
###############
# TensorBaord #
###############
class TensorBoardRunner:
    def __init__(self, path, host, port):
        logger.info('[TENSORBOARD] Start TensorBoard process!')
        self.server = TensorboardServer(path, host, port)
        self.server.start()
        self.daemon = True
         
    def finalize(self):
        if self.server.is_alive():    
            self.server.terminate()
            self.server.join()
        logger.info('[TENSORBOARD] ...finished TensorBoard process!')
        
    def interrupt(self):
        self.server.pkill()
        if self.server.is_alive():    
            self.server.terminate()
            self.server.join()
        logger.info('[TENSORBOARD] ...interrupted; killed all TensorBoard processes!')

class TensorboardServer(Process):
    def __init__(self, path, host, port):
        super().__init__()
        self.os_name = os.name
        self.path = str(path)
        self.host = host
        self.port = port
        self.daemon = True

    def run(self):
        if self.os_name == 'nt':  # Windows
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.path}" --host {self.host} --port {self.port} 2> NUL')
        elif self.os_name == 'posix':  # Linux
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.path}" --host {self.host} --port {self.port} >/dev/null 2>&1')
        else:
            err = f'Current OS ({self.os_name}) is not supported!'
            logger.exception(err)
            raise Exception(err)
    
    def pkill(self):
        if self.os_name == 'nt':
            os.system(f'taskkill /IM "tensorboard.exe" /F')
        elif self.os_name == 'posix':
            os.system('pgrep -f tensorboard | xargs kill -9')