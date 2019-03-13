# -*- coding: utf-8 -*-

import os
import sys
import logging
import random

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from deepwalk import graph
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

def process(args):
  if args.format == "adjlist":
    G = graph.load_adjacencylist(args.input, undirected=args.undirected)
  else:
    raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)

  print("Number of nodes: {}".format(len(G.nodes())))

  num_walks = len(G.nodes()) * args.number_walks
  print("Number of walks: {}".format(num_walks))

  data_size = num_walks * args.walk_length
  print("Data size (walks*length): {}".format(data_size))

  # Start walking
  print("Random walking...")
  walk_seq = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                      path_length=args.walk_length)
  print("Training...")
  model = Word2Vec(walk_seq, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1,
                   workers=args.workers)

  print("Training completed!")
  model.wv.save_word2vec_format(args.output)

def main():
  parser = ArgumentParser("deepwalk",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')

  parser.add_argument('--input', nargs='?', required=True,
                      help='Input graph file')

  parser.add_argument('--format', default='adjlist',
                      help='File format of input file')

  parser.add_argument('--undirected', default=True, type=bool,
                      help='Treat graph as undirected.')

  parser.add_argument('--number-walks', default=10, type=int,
                      help='Number of random walks to start at each node')

  parser.add_argument('--output', required=True,
                      help='Output representation file')

  parser.add_argument('--representation-size', default=64, type=int,
                      help='Number of latent dimensions to learn for each node.')

  parser.add_argument("-l", "--log", dest="log", default="INFO",
                      help="log verbosity level")

  parser.add_argument('--seed', default=0, type=int,
                      help='Seed for random walk generator.')

  parser.add_argument('--walk-length', default=40, type=int,
                      help='Length of the random walk started at each node')

  parser.add_argument('--window-size', default=5, type=int,
                      help='Window size of skipgram model.')

  parser.add_argument('--workers', default=1, type=int,
                      help='Number of parallel processes.')

  args = parser.parse_args()
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  process(args)

if __name__ == "__main__":
  sys.exit(main())
