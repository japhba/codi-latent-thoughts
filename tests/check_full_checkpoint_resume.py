"""GPU regression: recover the same maps at the last-row/counter boundary."""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from codi_repro.all_layers import fit


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True, help='Completed local capture/fit directory.')
parser.add_argument('--out', type=Path, required=True, help='New local regression directory.')
parser.add_argument('--shared', type=Path, required=True, help='New shared regression directory.')
parser.add_argument('--cache-dir', type=Path, required=True)
parser.add_argument('--organism-dir', type=Path)
parser.add_argument('--group', default='jl4')
args = parser.parse_args()
os.environ['WANDB_MODE'] = 'offline'
args.out.mkdir(parents=True); args.shared.mkdir(parents=True)
for name in ['fit_problems.json', 'validation_problems.json', 'fit_activations.npy', 'validation_activations.npy']: shutil.copyfile(args.source / name, args.out / name)
state = torch.load(args.source / 'layer_lens_resume.pt', map_location='cpu', weights_only=True)
assert state['row_cursor'] == 0 and state['finished']
n, args.dim_batch = state['count'], state['dim_batch']
assert state['history'][-1]['n_fit'] == n
state['count'] = n - 1; state['row_cursor'] = state['sums'].shape[1]; state['finished'] = False; state['history'] = state['history'][:-1]
del state['stop_reason']
torch.save(state, args.shared / 'layer_lens_resume.pt')
del state
args.resume = True; args.checkpoint_seconds = 120; args.eval_every = 8
fit(args)
original = torch.load(args.source / 'layer_jacobians.pt', map_location='cpu', mmap=True, weights_only=True)
resumed = torch.load(args.out / 'layer_jacobians.pt', map_location='cpu', mmap=True, weights_only=True)
assert original['meta']['n_problems'] == resumed['meta']['n_problems'] == n
assert torch.equal(original['J_emb'], resumed['J_emb'])
assert all(torch.equal(original['J'][layer], resumed['J'][layer]) for layer in original['J'])
result = {'test': 'Resume after all rows of the final example, before completed-example counter update', 'calibration_n': n, 'identical_maps': len(original['J']) + 1, 'all_equal': True}
(args.shared / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(result)
