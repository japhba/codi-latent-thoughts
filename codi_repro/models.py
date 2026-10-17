import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

from .analysis import ROOT, read


def prepare_cache(path):
    load_dotenv(Path.home() / '.env')
    load_dotenv()
    path.mkdir(parents=True, exist_ok=True)
    os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'
    return path


def fingerprint(path, expected):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): digest.update(block)
    assert digest.hexdigest() == expected, f'Wrong checkpoint: {path}'


def snapshot(spec, cache):
    from huggingface_hub import snapshot_download
    patterns = [spec['subfolder'] + '/*'] if 'subfolder' in spec else None
    return Path(snapshot_download(repo_id=spec['repo_id'], revision=spec['revision'], token=True, cache_dir=cache / 'hub', allow_patterns=patterns))


def load_organism(args):
    import torch
    prepare_cache(args.cache_dir)
    manifest = read('manifest.json')
    checkpoint = args.organism_dir if args.organism_dir else snapshot(manifest['organism'], args.cache_dir)
    if args.organism_dir: fingerprint(checkpoint / 'pytorch_model.bin', manifest['organism']['weights_sha256'])
    from .organism import load
    torch.manual_seed(11)
    model = load(checkpoint, read('artifacts/base_config.json'))
    model.tokenizer.bot_id = model.tokenizer.convert_tokens_to_ids('<|bocot|>')
    model.tokenizer.eot_id = model.tokenizer.convert_tokens_to_ids('<|eocot|>')
    assert model.codi.config.hidden_size == 4096 and len(model.codi.get_base_model().model.layers) == 36
    return model.eval()


def load_reader(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    prepare_cache(args.cache_dir)
    manifest = read('manifest.json')
    assert args.stream in manifest['reader_streams']
    assert args.reader_dir is not None, 'Supply the published final reader adapter directory.'
    fingerprint(args.reader_dir / 'adapter_model.safetensors', manifest['readers'][args.stream]['weights_sha256'])
    base_dir = args.reader_base_dir if args.reader_base_dir else snapshot(manifest['reader_base'], args.cache_dir) / 'av'
    if args.reader_base_dir:
        for name, digest in manifest['reader_base']['local_weights_sha256'].items(): fingerprint(base_dir / name, digest)
        for name, digest in manifest['reader_base']['local_metadata_sha256'].items(): fingerprint(base_dir / name, digest)
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model = PeftModel.from_pretrained(base, args.reader_dir).eval()
    return model, tokenizer
