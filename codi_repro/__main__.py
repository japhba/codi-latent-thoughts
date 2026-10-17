import argparse
from pathlib import Path

from .analysis import ROOT, figures, reproduce, verify

parser = argparse.ArgumentParser(description="Reproduce What's in CODI's latent thoughts?")
sub = parser.add_subparsers(dest='command', required=True)
sub.add_parser('reproduce', help='Offline: validate records, write statistics to results/, draw every post figure into figs/.')
sub.add_parser('figures', help='Offline: redraw every post figure into figs/ (fig_07 reads results/ from reproduce).')
sub.add_parser('validate', help='Check every bundled artifact and measurement schema.')
p = sub.add_parser('rerun', help='GPU: rerun the fixed-checkpoint measurements.')
p.add_argument('experiment', choices=['controls', 'twins', 'readouts', 'capture'])
p.add_argument('--out', type=Path, required=True)
p.add_argument('--cache-dir', type=Path, required=True)
p.add_argument('--organism-dir', type=Path)
p.add_argument('--reader-dir', type=Path)
p.add_argument('--reader-base-dir', type=Path)
p.add_argument('--stream', choices=['z', 'h24'])
p.add_argument('--batch-size', type=int, help='Default: the historical batch size recorded in manifest.json.')
p.add_argument('--limit', type=int)
p = sub.add_parser('judge', help='Optional paid API rerun of the published value-match criterion.')
p.add_argument('--readouts', type=Path, default=ROOT / 'artifacts/readouts.json')
p.add_argument('--out', type=Path, required=True)
p.add_argument('--concurrency', type=int, default=4)
p = sub.add_parser('lens', help='GPU: fixed-checkpoint LogitLens and Jacobian lens.')
p.add_argument('action', choices=['readouts', 'capture', 'fit', 'audit'])
p.add_argument('--out', type=Path, required=True)
p.add_argument('--cache-dir', type=Path, required=True)
p.add_argument('--organism-dir', type=Path)
p.add_argument('--base-dir', type=Path)
p.add_argument('--acts', type=Path)
p.add_argument('--lens', type=Path)
p.add_argument('--batch-size', type=int, default=256)
p.add_argument('--limit', type=int)
p.add_argument('--dim-batch', type=int, default=1)
p.add_argument('--n-fit', type=int, default=96)
p.add_argument('--resume', action='store_true')
p.add_argument('--recorded', action='store_true', help='Require the exact recovered activation and Jacobian fingerprints.')
p.add_argument('--problems', type=Path, default=ROOT / 'artifacts/lens_problems.json')
p = sub.add_parser('lens-analyze', help='Offline: score saved lens readouts and draw figures.')
p.add_argument('--inputs', type=Path, default=ROOT / 'artifacts')
p.add_argument('--out', type=Path, default=ROOT / 'results/lens')
p = sub.add_parser('av-lens', help='GPU: select and evaluate LogitLens baselines on the reader cohorts.')
p.add_argument('--out', type=Path, required=True)
p.add_argument('--cache-dir', type=Path, required=True)
p.add_argument('--organism-dir', type=Path)
p.add_argument('--batch-size', type=int, default=2)
p.add_argument('--readout-batch-size', type=int, default=64)
p = sub.add_parser('curve-lens', help='Complete-value LogitLens/J-lens references for the original post curves.')
p.add_argument('action', choices=['capture', 'readouts', 'judge'])
p.add_argument('--out', type=Path, required=True)
p.add_argument('--cache-dir', type=Path)
p.add_argument('--organism-dir', type=Path)
p.add_argument('--batch-size', type=int, default=2)
p.add_argument('--readout-batch-size', type=int, default=64)
p.add_argument('--lens', type=Path)
p.add_argument('--vectors', type=Path, default=ROOT / 'artifacts/curve_lens_ao_ensemble_vectors.npz')
p.add_argument('--resume', action='store_true', help='Resume saved pooled value extractions in --out.')
p.add_argument('--concurrency', type=int, default=16)
args = parser.parse_args()
if args.command == 'reproduce': reproduce()
elif args.command == 'figures': figures()
elif args.command == 'validate': print(verify())
elif args.command == 'rerun':
    from .experiments import run
    run(args)
elif args.command == 'lens':
    from .lenses import run
    run(args)
elif args.command == 'lens-analyze':
    from .lens_analysis import reproduce as reproduce_lenses
    args.out.mkdir(parents=True, exist_ok=True)
    reproduce_lenses(args.out, inputs=args.inputs)
elif args.command == 'av-lens':
    from .av_lens import run
    run(args)
elif args.command == 'curve-lens':
    from .curve_lenses import run
    run(args)
else:
    from .judge import run
    run(args)
