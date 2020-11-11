import os

import json
import numpy as np

from utils.evaluate import evaluate
from utils.helper import JsonEncoder

res_dir = 'run'
models = ('dcrnn', 'fclstm', 'gwnet', 'stgcn')

unified_results = {'speed': dict(), 'available': dict()}

for model in models:
    results = np.load(os.path.join(res_dir, model, 'test-results.npz'))
    predictions, targets = results['predictions'], results['targets']

    unified_results['speed'].update({model: evaluate(predictions[:, :, 0], targets[:, :, 0])})
    unified_results['available'].update({model: evaluate(predictions[:, :, 1], targets[:, :, 1])})

ha_res = json.load(open('run/ha_results.json'))
unified_results['speed'].update({'ha': ha_res['speed']})
unified_results['available'].update({'ha': ha_res['available']})

var_res = json.load(open('run/var_results.json'))
unified_results['speed'].update({'var': var_res['speed']})
unified_results['available'].update({'var': var_res['available']})

speed_results = np.load(os.path.join(res_dir, 'ours_speed', 'test-results.npz'))
avail_results = np.load(os.path.join(res_dir, 'ours_avail', 'test-results.npz'))
unified_results['speed'].update({'sp': evaluate(speed_results['predictions'], speed_results['targets'])})
unified_results['available'].update({'ap': evaluate(avail_results['predictions'], avail_results['targets'])})

print(json.dumps(unified_results, indent=4, cls=JsonEncoder))
json.dump(unified_results, open('data/results.json', 'w+'), indent=4, cls=JsonEncoder)
