import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

ranks = np.array([1, 2, 4, 8, 16])

runtime_S = np.array([4.04, 1.89, 1.84, 1.51, 1.59])
runtime_M = np.array([48.51, 21.25, 20.43, 15.21, 15.54])
runtime_L = np.array([288.15, 107.84, 103.14, 92.39, 92.19])

speedup_S = runtime_S[0] / runtime_S
speedup_M = runtime_M[0] / runtime_M
speedup_L = runtime_L[0] / runtime_L

eff_S = speedup_S / ranks
eff_M = speedup_M / ranks
eff_L = speedup_L / ranks

datasets = [('S2k', runtime_S), ('M5k', runtime_M), ('L10k', runtime_L)]
for i, (name, rt) in enumerate(datasets):
    plt.figure()
    plt.plot(ranks, rt, marker='o')
    plt.xscale('log', basex=2)
    plt.xlabel('#Ranks')
    plt.ylabel('Runtime (s)')
    plt.title(f'Runtime vs Ranks - {name}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'fig_C{i+1}_{name}.png', dpi=300, bbox_inches='tight')
    plt.close()

plt.figure()
plt.plot(ranks, speedup_S, marker='o', label='S2k')
plt.plot(ranks, speedup_M, marker='o', label='M5k')
plt.plot(ranks, speedup_L, marker='o', label='L10k')
plt.plot(ranks, ranks, 'k--', label='Ideal')
plt.xscale('log', basex=2)
plt.yscale('log', basey=2)
plt.xlabel('#Ranks')
plt.ylabel('Speedup')
plt.title('Strong Scaling Speedup')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('fig_A_speedup.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(ranks, eff_S, marker='o', label='S2k')
plt.plot(ranks, eff_M, marker='o', label='M5k')
plt.plot(ranks, eff_L, marker='o', label='L10k')
plt.xscale('log', basex=2)
plt.xlabel('#Ranks')
plt.ylabel('Efficiency')
plt.title('Parallel Efficiency')
plt.ylim(0, None)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('fig_B_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot([2000, 5000, 10000],
         [speedup_S[-1], speedup_M[-1], speedup_L[-1]],
         marker='o')
plt.xlabel('Graph Size (nodes)')
plt.ylabel('Speedup @ 16 ranks')
plt.title('Scaling Benefit with Problem Size')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('fig_D_scaling_benefit.png', dpi=300, bbox_inches='tight')
plt.close()
