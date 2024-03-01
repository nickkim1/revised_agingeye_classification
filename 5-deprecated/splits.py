import pandas as pd
import argparse as ap
import numpy as np 

parser = ap.ArgumentParser(prog='This program processes sequences generated from ATAC-seq data using bedtools getfasta')
parser.add_argument('-csvfile', type=str, help='Input bed file filepath used to generate fasta file', nargs=1)
args = vars(parser.parse_args())

csv_file = args['csvfile'][0]
