# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')

import h5py
import matplotlib.pyplot as plt

from ..features._mp import calc_main_parameters

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('pl_channel', metavar='0', type=int, nargs='+',
                    help='Select phonon (usually 0) or light(usually 1) channel.')
parser.add_argument('index', metavar='0', type=int, nargs='+',
                    help='Integer to plot corresponding event.')
parser.add_argument('fpath', metavar='', type=str, nargs='+',
                    help='Filepath to h5-file.')
parser.add_argument('-T', type=str, nargs=1, help='Title of plot.')
parser.add_argument('-m', action='store_true', help='Add mainparameters to the plot.')

args = parser.parse_args()
pl_channel = args.pl_channel[0]
index = args.index[0]
fpath = args.fpath[0]

path = '../data/'
path_plots = './plots'

if not os.path.exists(path_plots):
    os.makedirs(path_plots)

ds = h5py.File(fpath, 'r')
event = ds['events/event'][pl_channel,index,:]
if args.m:
    event_mainpar = calc_main_parameters(event)

plt.close()
plt.figure()
# plt.xlim((0,event.shape[0]))


plt.plot(event, linewidth=1, zorder=-1)
if args.m:
    event_mainpar.plotParameters()

plt.ylim(plt.ylim())

if args.m:
    #offset line
    plt.hlines(y=event_mainpar.offset, xmin=plt.xlim()[0], xmax=plt.xlim()[1], linestyles='--', alpha=0.2)
# 1/4 Line approx point of trigger
plt.vlines(x=int(event.shape[0]/4), ymin=plt.ylim()[0], ymax=plt.ylim()[1], linestyles='--', alpha=0.2)

save = True

if args.T:
    plt.title(args.T[0])

plt.tight_layout()

if args.T:
    if save:
        plt.savefig('{}/{}-{}-{}-{}.pdf'.format(path_plots,fpath.split('/')[-1].split('-')[0],
                                                index,
                                                pl_channel,
                                                ds['events/labels'][pl_channel,index]))

# import ipdb; ipdb.set_trace()
plt.show()
