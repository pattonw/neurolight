import os
from subprocess import check_call
from funlib.run import run, run_singularity
import logging
import scipy.interpolate.interpnd  # Workaround for cython __reduce_cython__ error
import sys
from config_parser import (
    read_worker_config,
    read_data_config,
    read_graph_config,
    read_solve_config,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
out_hdlr.setLevel(logging.DEBUG)
logger.addHandler(out_hdlr)
logger.setLevel(logging.DEBUG)

config_file = "config_full.ini"
directory = os.path.dirname(os.path.abspath(__file__))
predict_config_path = os.path.join(directory, config_file)
worker_config_path = os.path.join(directory, config_file)
data_config_path = os.path.join(directory, config_file)
graph_config_path = os.path.join(directory, config_file)
solve_config_path = os.path.join(directory, config_file)

solve_config = read_solve_config(solve_config_path)

base_cmd = "python {} {} {} {} {} {}".format(
    solve_config["daisy_solve"],
    predict_config_path,
    worker_config_path,
    data_config_path,
    graph_config_path,
    solve_config_path,
)

check_call(
    base_cmd,
    shell=True,
    stdout=open(os.path.join(directory, "solve.out"), "w"),
    stderr=open(os.path.join(directory, "solve.err"), "w"),
)

