import sys

import yaml

from functions.generic import BColours

##############################################################################
# I/O functions for parameters.yml file
##############################################################################

def load_parameters(yaml_file):
    variables = {}

    try:
        with open(yaml_file, "r") as f:
            nested_variables = yaml.safe_load(f)
    except Exception as e:
        print(f"{BColours.FAIL}Error with parameters.yml file: {e}..{BColours.ENDC}")
        sys.exit(0)
    else:
        # Remove nested dictionary from config_variables
        for parameters in nested_variables.values():
            for k,v in parameters.items():
                variables[k] = v

    return variables