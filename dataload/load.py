import os
import argparse
import logging
from gen import TPCHGenerator, Connection, Replica

def get_replicas(path: str):
    replicas = []
    with open(path, 'r') as infile:
        lines = infile.readlines()
        for config in lines:
            fields = config.split(',')
            replicas.append(
                Replica(
                    id=fields[0],
                    hostname=fields[1],
                    port=fields[2],
                    dbname=fields[3],
                    user=fields[4],
                    password=fields[5]
                )
            )
    return replicas

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data-dir', type=str, default='/pgdata', help='where the data root is')
    parser.add_argument('-s', '--scale-factor', type=int, help='the scale factor')
    parser.add_argument('-r', '--replica-path', type=str, default='../replicas.csv', help='the location of replicas.csv')
    parser.add_argument('-v', '--verbose', action='store_true')

    return parser.parse_args()

def initiate_load(args):
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    base = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base, args.data_dir)
    data_dir = os.path.normpath(data_dir)

    replicas = get_replicas(args.replica_path)

    generator = TPCHGenerator(replicas, data_dir, args.scale_factor)

    logging.info('!! starting table load, scale factor', args.scale_factor)

    generator.load_database()

if __name__ == '__main__':
    args = parse_arguments()

    initiate_load(args)
