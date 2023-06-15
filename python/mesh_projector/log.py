#
# This file is part of the CAD-ASTRA distribution (git@github.com:cad-astra/cad-astra.git).
# Copyright (c) 2021-2023 imec-Vision Lab, University of Antwerp.
#
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
Setting up Python global logging object.
"""

import logging

levels_dict = {
    "debug":    logging.DEBUG,
    "info":     logging.INFO,
    "warning":  logging.WARNING,
    "error":    logging.ERROR,
    "critical": logging.CRITICAL
}

optix_levels_dict = {
    "disable" : 0,
    "fatal"   : 1,
    "error"   : 2,
    "warning" : 3,
    "print"   : 4
}

def main_logger_name():
    return "mesh-fp-main"

def setup_logger(name="mesh-fp-logger"):
    formatter = logging.Formatter(fmt='[%(asctime)s] - %(levelname)s - %(module)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    logger.info(f"Logger '{name}' is configured")