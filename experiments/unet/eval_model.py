import argparse
import yaml
import os

from jsonmerge import merge 

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--configs', type=str, nargs='+', required=True, help='Configs for model training')
    args = parser.parse_args()
    
    config = {}
    for conf in args.configs:
        with open(conf, 'r') as stream:
            config = merge(config, yaml.safe_load(stream))
    
    if config['model_type'] == 'full_unet':
        from unet.evaluator.full_unet_evaluator import FullUnetEvaluator
        trainer = FullUnetEvaluator(config)
        trainer.evaluate()
    else:
        raise Exception(f'No trainer found for model type {config["model_type"]}') 