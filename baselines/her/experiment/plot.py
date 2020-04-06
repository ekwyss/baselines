# DEPRECATED, use baselines.common.plot_util instead

import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns; sns.set()
import glob2
import argparse
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')


def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    return xsmoo, ysmoo


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)

        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


parser = argparse.ArgumentParser()
# parser.add_argument('filedir1', type=str)
parser.add_argument('-filedirs1', nargs='+', type=str)
# parser.add_argument('filedir2', type=str)
# parser.add_argument('-filedirs2', nargs='+', type=str)
parser.add_argument('-num_epochs', type=int)
# parser.add_argument('-num_epochs', type=int)
parser.add_argument('-outfile', type=str)
args = parser.parse_args()
# filedir1 = "/home/eric/baselines/policies/her500000Output"
# filedir2 = "/home/eric/baselines/policies/150000"
# filedir2 = "/home/eric/baselines/policies/217newsingle2"
# filedir2 = "/home/eric/openai-2020-02-11-22-37-24-858984"
# filedir2 = "/home/eric/baselines/policies/100000"

def make_data(filedir):
    # Load all data.
    data = {}
    paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(filedir, '**', 'progress.csv'))]
    # paths = []
    # for filedir in filedirs:
    #     paths.append(os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(filedir, '**', 'progress.csv')))

    for curr_path in paths:
        if not os.path.isdir(curr_path):
            continue
        results = load_results(os.path.join(curr_path, 'progress.csv'))
        if not results:
            print('skipping {}'.format(curr_path))
            continue
        print('loading {} ({})'.format(curr_path, len(results['epoch'])))
        with open(os.path.join(curr_path, 'params.json'), 'r') as f:
            params = json.load(f)

        success_rate = np.array(results['test/success_rate'])
        epoch = np.array(results['epoch']) + 1
        env_id = params['env_name']
        replay_strategy = params['replay_strategy']

        if replay_strategy == 'future':
            config = 'her'
        else:
            config = 'ddpg'
        if 'Dense' in env_id:
            config += '-dense'
        else:
            config += '-sparse'
        env_id = env_id.replace('Dense', '')

        # Process and smooth data.
        assert success_rate.shape == epoch.shape
        x = epoch
        y = success_rate
        # if args.smooth:
        if 1:
            x, y = smooth_reward_curve(epoch, success_rate)
        assert x.shape == y.shape

        if env_id not in data:
            data[env_id] = {}
        if config not in data[env_id]:
            data[env_id][config] = []
        data[env_id][config].append((x, y))
    return data

# #get results from all "results<i>" folders within specified folder
# success_rates1 = []
# for filedir in os.listdir(args.filedir1):
#     if filedir[:7] == 'results':
#         full_path = os.path.join(args.filedir1, filedir)
#         data1 = make_data(full_path)
#         if len(data1['FetchPickAndPlace-v1']['her-sparse'][0][0]) >= 40:
#             success_rates1.append(data1['FetchPickAndPlace-v1']['her-sparse'][0][1])
# data1['FetchPickAndPlace-v1']['her-sparse'][0] = (data1['FetchPickAndPlace-v1']['her-sparse'][0][0], np.mean(success_rates1, axis=0))
# datas = [data1]
# labels = [args.filedir1.split("/")[-3] + "/" + args.filedir1.split("/")[-2]]

# if args.filedir2 != "NA":
#     success_rates2 = []
#     for filedir in os.listdir(args.filedir2):
#         if filedir[:7] == 'results':
#             full_path = os.path.join(args.filedir2, filedir)
#             data2 = make_data(full_path)
#             if len(data2['FetchPickAndPlace-v1']['her-sparse'][0][0]) >= 40:
#                 success_rates2.append(data2['FetchPickAndPlace-v1']['her-sparse'][0][1])
#     data2['FetchPickAndPlace-v1']['her-sparse'][0] = (data2['FetchPickAndPlace-v1']['her-sparse'][0][0],np.mean(success_rates2, axis=0))
#     data.append(data2)
#     labels.append(args.filedir2.split("/")[-3] + "/" + args.filedir2.split("/")[-2])

datas = []
labels = []
for filedir1 in args.filedirs1:
    success_rates1 = []
    for filedir in os.listdir(filedir1):
        if filedir[:7] == 'results':
            full_path = os.path.join(filedir1, filedir)
            data1 = make_data(full_path)
            # if len(data1['FetchPickAndPlace-v1']['her-sparse'][0][0]) >= 40:
            success_rates1.append(data1['FetchPickAndPlace-v1']['her-sparse'][0][1])
            # print(filedir, data1)
    data1['FetchPickAndPlace-v1']['her-sparse'][0] = (data1['FetchPickAndPlace-v1']['her-sparse'][0][0], np.mean(success_rates1, axis=0))
    datas.append(data1)
    labels.append([filedir1.split("/")[-2]])
    # labels.append([filedir1.split("/")[-3] + "/" + filedir1.split("/")[-2]])

#If passed in all relevant folders on command line
# success_rates1 = []
# for filedir in args.filedirs1:
#     data1 = make_data(filedir)
#     if len(data1['FetchPickAndPlace-v1']['her-sparse'][0][0]) == 100:
#         success_rates1.append(data1['FetchPickAndPlace-v1']['her-sparse'][0][1])

# success_rates2 = []
# for filedir in args.filedirs2:
#     data2 = make_data(filedir)
#     if len(data2['FetchPickAndPlace-v1']['her-sparse'][0][0]) == 100:
#         success_rates2.append(data2['FetchPickAndPlace-v1']['her-sparse'][0][1])
# print(data1)
# print(np.mean(data1, axis=0))
# print(10/0)
# data1 = make_data(args.filedir1)
# data2 = make_data(args.filedir2)

# print(data2)
# print(type(data2))
# print(data2.keys())
# print(data2['FetchPickAndPlace-v1'])
# print(type(data2['FetchPickAndPlace-v1']))
# print(data2['FetchPickAndPlace-v1'].keys())
# print(data2['FetchPickAndPlace-v1']['her-sparse'])
# print(type(data2['FetchPickAndPlace-v1']['her-sparse']))
# print(data2['FetchPickAndPlace-v1']['her-sparse'][0])
# print(type(data2['FetchPickAndPlace-v1']['her-sparse'][0]))


# datas = [np.mean(data1, axis=0), np.mean(data2, axis=0)]
# datas = [data1, data2]
# datas = [data1, data2]
# labels = ["Original HER", "Subgoal Based HER"]
# Plot data.
for env_id in sorted(data1.keys()):
    print('exporting {}'.format(env_id))
    plt.clf()

    for data,label in zip(datas,labels):
        for config in sorted(data[env_id].keys()):
            xs, ys = zip(*data[env_id][config])
            xs, ys = pad(xs), pad(ys)
            print(xs.shape)
            print(ys.shape)
            assert xs.shape == ys.shape
            xnew = []
            ynew = []
            for i in range(len(xs)):
                xnew.append(xs[i][:args.num_epochs])
                ynew.append(ys[i][:args.num_epochs])
            xs = np.array(xnew)
            ys = np.array(ynew)

            plt.plot(xs[0], np.nanmedian(ys, axis=0), label=label)
            plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25)
    plt.title(env_id)
    plt.xlabel('Epoch')
    plt.ylabel('Median Success Rate')
    plt.ylim(0.0,1.0)
    plt.legend(loc='lower right', prop=fontP)
    # plt.savefig(os.path.join('/home/eric/baselines/policies', 'fig_{}.png'.format(env_id)))
    # if args.filedir2 != "NA":
    #     plt.savefig(os.path.join(args.filedir2, 'fig_{}.png'.format(env_id)))
    # else:
    #     plt.savefig(os.path.join(args.filedir1, 'fig_{}.png'.format(env_id)))

    # if len(args.filedirs1) == 1:
    #     plt.savefig(os.path.join(args.filedirs1[0], 'fig_{}.png'.format(env_id)))
    # else:
    #     file_ending = ""
    #     for filedir in args.filedirs1:
    #         file_ending += filedir.split("/")[-3] + "_" + filedir1.split("/")[-2] + "_"
    #     plt.savefig(os.path.join("/home/eric/baselines/policies/plots/", 'fig_{}_{}.png'.format(env_id, file_ending)))
    plt.savefig(args.outfile)

 # python3.6 baselines/baselines/her/experiment/plot.py -filedirs1 /home/eric/baselines/policies/goalind/5_5_20_nodemo_nosgrewinER/  /home/eric/baselines/policies/goalind/5_5_20_1demo_nosgrewinER/  /home/eric/baselines/policies/goalind/5_5_20_10demo_nosgrewinER/ /home/eric/baselines/policies/goalind/5_5_20_100demo_nosgrewinER/ /home/eric/baselines/policies/original_her/100demo/ /home/eric/baselines/policies/original_her/nodemo/ -num_epochs 100 -outfile /home/eric/baselines/policies/plots/GripConstraint_5_5_20_noERsgrew_comp.png
