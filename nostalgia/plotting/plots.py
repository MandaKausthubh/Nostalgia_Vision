from typing import List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_accuracy_vs_task(acc_per_task: List[float], out: str):
    plt.figure()
    plt.plot(range(1, len(acc_per_task)+1), acc_per_task, marker='o')
    plt.xlabel('Task Index')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Task Index')
    plt.grid(True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches='tight'); plt.close()


def plot_forgetting_curve(forgetting: List[float], out: str):
    plt.figure()
    plt.plot(range(1, len(forgetting)+1), forgetting, marker='o', color='r')
    plt.xlabel('Task Index')
    plt.ylabel('Forgetting')
    plt.title('Forgetting Measure over Tasks')
    plt.grid(True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches='tight'); plt.close()


def plot_avg_accuracy(avg_acc: float, out: str):
    plt.figure()
    plt.bar(["Avg Acc"], [avg_acc])
    plt.ylim(0, 1)
    plt.title('Average Accuracy Over Tasks')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches='tight'); plt.close()


def plot_training_loss_curve(log_csv: str, out: str, loss_col: str = 'train/loss', step_col: str = 'step'):
    df = pd.read_csv(log_csv)
    plt.figure()
    sns.lineplot(data=df, x=step_col, y=loss_col)
    plt.title('Training Loss Curve')
    plt.grid(True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches='tight'); plt.close()


def plot_memory_overhead(mem_records_csv: str, out: str):
    df = pd.read_csv(mem_records_csv)
    plt.figure()
    sns.lineplot(data=df, x='step', y='gpu_mem_mb', label='GPU Mem (MB)')
    sns.lineplot(data=df, x='step', y='cpu_mem_mb', label='CPU Mem (MB)')
    plt.title('Memory/Computation Overhead')
    plt.grid(True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches='tight'); plt.close()


def plot_param_importance(importance_csv: str, out: str):
    df = pd.read_csv(importance_csv).sort_values('importance', ascending=False).head(50)
    plt.figure(figsize=(8, 10))
    sns.barplot(data=df, x='importance', y='param')
    plt.title('Top Parameter Importances')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches='tight'); plt.close()
