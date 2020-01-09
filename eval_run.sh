export PYTHONPATH=$PYTHONPATH:$PWD
for i in $(seq 0 8)
do
echo $i
CUDA_VISIBLE_DEVICES=2,1,0 python FastAutoAugment/train.py -c confs/pyramid272_cifar100_2_tl.yaml --aug fa_reduced_cifar10 --dataset permutated_cifar100 --save cifar100_pyramid272_top1_11.74$i_e%d_top1_%.3f_%.3f.pth --nb-labels 2 --classifier-id $i --dataroot /home/zhuzby/data --only-eval
done