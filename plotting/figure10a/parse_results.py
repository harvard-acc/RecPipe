import os, sys
import csv
import numpy as np
import matplotlib.pyplot as plt

def extract_results(results_name):
    with open(results_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        line_count   = 0
        cycles       = []
        utilizations = []
        for row in csv_reader:
            if line_count != 0:
                cycles.append(float(row[1]))
                utilizations.append(float(row[2]))
            line_count +=1
            
        total_cycles      = np.sum(cycles)
        total_utilization = np.sum((np.array(cycles)/total_cycles)*np.array(utilizations))
        
        print('File: {}\nCycles: {}\nUtilization: {}%\n'.format(results_name, total_cycles, total_utilization))

def main():
    SYSTOLIC_DATA_DIR  = "./RPAccel_characterization/systolic_array/outputs/"

    results = sorted(os.listdir(SYSTOLIC_DATA_DIR))

    for result in results:
        # Uncomment conditional below to pritn results of all dataflows.
        # Note that in the paper we only present result of weight stationary dataflow.
        if result.endswith('ws'):
            result_cycles_file = [file for file in os.listdir(SYSTOLIC_DATA_DIR+result) if file.endswith('cycles.csv')][0]
            extract_results(SYSTOLIC_DATA_DIR+result+'/'+result_cycles_file)

if __name__ == "__main__":
    main()