import os
import json
from ml.repro import create_run_manifest


def test_create_manifest_basic():
    dataset = os.path.abspath('Train.csv')
    config = {'seed': 123, 'dataset': dataset}
    manifest = create_run_manifest([dataset], config, out_dir='runs_test')

    assert 'run_id' in manifest
    assert 'datasets' in manifest
    assert os.path.exists(os.path.join('runs_test', manifest['run_id'], 'manifest.json'))
    # dataset hash should be present (may be None if file missing)
    assert 'Train.csv' in manifest['datasets'] or os.path.basename(dataset) in manifest['datasets']


if __name__ == '__main__':
    test_create_manifest_basic()
    print('OK')
