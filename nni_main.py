import nni
import logging
from nni.utils import merge_parameter
from lt import get_argument_parser, main

logger = logging.getLogger("diabetes_nni")

if __name__ == "__main__":
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        args = get_argument_parser()
        params = merge_parameter(args, tuner_params)
        params.experiment_id = nni.get_experiment_id()
        params.trail_id = nni.get_trial_id()
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
