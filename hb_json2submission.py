import argparse
import csv
import logging
import os
import json

from hb_metric import eval_dicts_uuid, extract_eval_answers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Huggingface Bert json output to cs224n submission csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("hb_json", help="Huggingface Bert json")
    parser.add_argument("sub_csv", help="submission file")
    args = parser.parse_args()
    return args


args = parse_args()

NAME = 'convert'

formatter = logging.Formatter('%(asctime)s %(name)12s [%(levelname)s]\t%(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(ch)
logger = logging.getLogger(__name__)


def main(args):

    logger.info("Reading hb json...")
    with open(args.hb_json) as fin:
        sub_dict = json.load(fin)

    logger.info("Writing to submission csv...")
    with open(args.sub_csv, 'w') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])
    logger.info("Done")


if __name__ == "__main__":
    main(args)



