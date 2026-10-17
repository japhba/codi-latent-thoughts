"""Stage recovered evaluation weights on a compute node's local fast disk."""
import argparse
from pathlib import Path
import shutil

from tqdm.auto import tqdm


def copy_tree(source, target):
    files = [p for p in source.rglob('*') if p.is_file()]
    for path in tqdm(files, desc=source.name, leave=False):
        out = target / path.relative_to(source)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.is_file() and out.stat().st_size == path.stat().st_size: continue
        partial = out.with_name(out.name + '.partial')
        shutil.copy2(path, partial)
        partial.replace(out)


def main(args):
    copy_tree(args.project / 'checkpoints/nla_warmstart_nz1/av', args.cache / 'av')
    for stream in tqdm(['z', 'h24'], desc='Reader sources'):
        source = args.project / f'checkpoints/nla_nz4_{stream}'
        for step in tqdm(range(400, 3201, 400), desc=stream, leave=False):
            copy_tree(source / f'iter_{step:06d}', args.cache / stream / f'iter_{step:06d}')
        copy_tree(source / 'critic_latest', args.cache / stream / 'critic_latest')
        shutil.copy2(source / 'nla_meta.yaml', args.cache / stream / 'nla_meta.yaml')
    copy_tree(args.project / 'checkpoints/codi_qwen3-8b/checkpoint-7580', args.cache / 'organism')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    main(parser.parse_args())
