import os, sys
import time
import json

import numpy as np

import matplotlib.pyplot as plt

def generate_filename(filter_ratio, cache_capacity_total, cache_ratio, RESULTS_DIR):

    outfile = './{}/{}_{}_{}.json'.format(RESULTS_DIR, filter_ratio, int(cache_capacity_total/(1024**2)), cache_ratio)
    return outfile

def AMAT(hit_rate, sram_cycles, dram_cycles):
    return hit_rate * sram_cycles + (1 - hit_rate) * dram_cycles

def main():

    PLOTTING_DIR = './plotting/figure10c'
    RESULTS_DIR = 'RPAccel_characterization/embedding/results'

    # Setup 

    filter_ratios          = [0.25, 0.2, 0.125, 0.1, 0.0625]
    cache_capacities_total = np.array([1, 2, 4, 6, 8, 12, 16]) * (1024**2)
    cache_ratios           = np.array([0.0, 0.01, 0.03, 0.05, 0.07, 
                                       0.09, 0.1, 0.2, 0.3, 0.4, 
                                       0.5, 0.6, 0.7, 0.8, 0.9, 
                                       0.91, 0.93, 0.95, 0.97, 0.99, 1.0])

    # 4MB vs 12MB Analysis

    ratios_plot     = [filter_ratios[x] for x in [2,2,4]]
    capacities_plot = [cache_capacities_total[x] for x in [2,5,5]]
    labels          = ['4MB (1/8)', '12MB (1/8)', '12MB (1/16)']

    plt.figure(dpi=128)

    for i in range(len(ratios_plot)):
        filter_ratio = ratios_plot[i]
        capacity     = capacities_plot[i]
        sub_label    = labels[i]
        sram_ratios  = []
        for ratio in cache_ratios:
            filename = generate_filename(filter_ratio, capacity, ratio, RESULTS_DIR)
            results  = json.load(open(filename))
            sram_ratios.append(results['bytes_sram_total'] / results['bytes_total'])
                
        plt.plot(cache_ratios, AMAT(np.array(sram_ratios), sram_cycles = 1, dram_cycles = 100), label = sub_label)
        
    plt.xlim([0,1])
    plt.xlabel('Fraction of cache for first stage')
    plt.ylabel('AMAT (Cycles)')
    plt.title('Embedding Cache Partitioning')
    plt.legend()
    plt.savefig('{}/4_12mb_amat.png'.format(PLOTTING_DIR))
    plt.show()

if __name__ == "__main__":
    main()