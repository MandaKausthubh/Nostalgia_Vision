import argparse
import os
import json
import numpy as np
from nostalgia.metrics.retention import forgetting_curve, average_accuracy
from nostalgia.plotting.plots import (
    plot_accuracy_vs_task, plot_forgetting_curve, plot_avg_accuracy,
    plot_training_loss_curve, plot_memory_overhead, plot_param_importance,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', type=str, required=True)
    p.add_argument('--log_csv', type=str, default=None)
    p.add_argument('--mem_csv', type=str, default=None)
    p.add_argument('--importance_csv', type=str, default=None)
    args = p.parse_args()

    figs = os.path.join(args.results_dir, 'figures')
    os.makedirs(figs, exist_ok=True)

    # Expect a JSON with per-task accuracies, e.g., [0.7, 0.65, 0.62]
    acc_json = os.path.join(args.results_dir, 'per_task_acc.json')
    if os.path.exists(acc_json):
        acc = json.load(open(acc_json))
        plot_accuracy_vs_task(acc, os.path.join(figs, 'acc_vs_task.png'))
        forg = forgetting_curve(acc)
        plot_forgetting_curve(forg, os.path.join(figs, 'forgetting.png'))
        plot_avg_accuracy(float(np.mean(acc)), os.path.join(figs, 'avg_acc.png'))

    if args.log_csv and os.path.exists(args.log_csv):
        plot_training_loss_curve(args.log_csv, os.path.join(figs, 'train_loss.png'))

    if args.mem_csv and os.path.exists(args.mem_csv):
        plot_memory_overhead(args.mem_csv, os.path.join(figs, 'memory_overhead.png'))

    if args.importance_csv and os.path.exists(args.importance_csv):
        plot_param_importance(args.importance_csv, os.path.join(figs, 'param_importance.png'))


if __name__ == '__main__':
    main()
