#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_attack.py --color --verbose 1 --attack badnet --pretrained --validate_interval 1 --epochs 50 --lr 1e-2



import trojanvision
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    trojanvision.defenses.add_argument(parser)
    kwargs = parser.parse_args().__dict__

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)

    kwargs['pretrained'] = True
    benign_model = trojanvision.models.create(dataset=dataset, **kwargs)
    kwargs['pretrained'] = False
    model = trojanvision.models.create(dataset=dataset, **kwargs)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)

    mark = trojanvision.marks.create(dataset=dataset, **kwargs)
    attack = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **kwargs)
    kwargs['benign_model'] = benign_model
    defense = trojanvision.defenses.create(dataset=dataset, model=model, attack=attack, **kwargs)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack, defense=defense)
    defense.detect(**trainer)
