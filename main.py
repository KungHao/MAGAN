import argparse

from agents import MAGANAgent
from utils.config import *


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--config',
        default='D:/James/Jupyter/MAGAN/configs/configuration.yaml',
        help='The path of configuration file in yaml format')
    args = arg_parser.parse_args()
    config = process_config(args.config)
    agent = MAGANAgent(config)
    agent.run()

if __name__ == '__main__':
    main()

