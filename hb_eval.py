import argparse
import datetime
import logging
import os
import json

from hb_metric import eval_dicts_uuid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate perf from files by SQuAD metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-m", "--mode", help="squad or huggingface bert output", choices=["sq", "hb"], default="hb")
    parser.add_argument("gold", help="gold references")
    parser.add_argument("pred", help="predictions")

    args = parser.parse_args()
    return args


args = parse_args()

NAME = 'eval'
# LOGGER_NAME = '[{}]'.format(NAME)
# ROOT = '.'
# PROCESS_START_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

formatter = logging.Formatter('%(asctime)s %(name)12s [%(levelname)s]\t%(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
# fh = logging.FileHandler("{}_{}.log".format(NAME, PROCESS_START_TIME))
# fh.setFormatter(formatter)

logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(ch)
# logging.getLogger().addHandler(fh)

logger = logging.getLogger(__name__)


def main(args):

    # Load predictions
    if args.mode == "hb":
        with open(args.pred) as fin:
            preds = json.load(fin)
    elif args.mode == "sq":
        preds = {}
        with open(args.pred) as fin:
            next(fin)
            for line in fin:
                line = line.rstrip(os.linesep)
                idx = line.index(",")
                uuid = line[:idx]
                pred = line[idx+1:]
                preds[uuid] = pred
    else:
        raise ValueError("Impossible mode")

    # Load gold references
    with open(args.gold) as fin:
        golds = json.load(fin)

    evals = eval_dicts_uuid(golds, preds)
    logger.info("EM, F1, AvNA: {},{},{}".format(evals["EM"], evals["F1"], evals["AvNA"]))


if __name__ == "__main__":
    main(args)
