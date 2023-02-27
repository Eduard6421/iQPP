import argparse
import DataLoaders.Loaders



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,help='Dataset on which you want to train the model',choices=['roxford5k','rparis6k','pascalvoc_700_medium','caltech101_700'])
parser.add_argument('--model', required=True,help='Dataset on which you want to train the model',choices=['masked','denoising'])
parser.add_argument('--mode', required=True,help='train/run', choices=['train','run'])
args = vars(parser.parse_args())

if(args['dataset'] not in ['roxford5k','rparis6k','pascalvoc_700_medium','caltech101_700']):
    raise Exception('unkown dataset')

if(args['mode'] not in ['train','run']):
    raise Exception('Unkown')


if(args['mode'] == 'train'):
    if(args['model'] == 'denoising'):
        import Train.train_dae
    else:
        import Train.train_mae
else:
    if(args['model'] == 'denoising'):
        import Run.run_dae
    else:
        import Run.run_mae
