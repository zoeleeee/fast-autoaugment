export PYTHONPATH=$PYTHONPATH:$PWD
python FastAutoAugment/train.py -c confs/pyramid272_cifar100_2.yaml --aug fa_reduced_cifar10 --dataset permutated_cifar100 --save permutated_cifar100_2.pth