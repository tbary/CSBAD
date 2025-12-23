import csv
from pathlib import Path

def write_results_csv(csv_path : str, 
                      dataset : str, 
                      n_1 : int, 
                      n_2 : int, 
                      select : str, 
                      filter : str,
                      embs : str, 
                      epochs:int, 
                      results_dict):
    """
    csv_path: path to the CSV file
    dataset: e.g. 'cifar10'
    n_samples: e.g. 1000 (your train subset size)
    filtering_strategy: e.g. 'first_n'
    results_dict: e.g. {'metrics/accuracy_top1': 0.20, 'metrics/accuracy_top5': 0.67, 'fitness': 0.43}
    """
    csv_path = Path(csv_path)
    metric_cols = list(results_dict.keys())          # keeps insertion order
    header = ['dataset', 'n_1', 'n_2', 'select', 'filter', 'embs', 'epochs'] + metric_cols 

    write_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open('a', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='|')
        if write_header:
            w.writerow(header)
        row = [dataset, n_1, n_2, select, filter, embs, epochs] + [results_dict[k] for k in metric_cols] #+ [config_dict[k] for k in config_dict]
        w.writerow(row)
