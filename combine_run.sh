for i in $(seq 1 10)
echo $i
do
CUDA_VISIBLE_DEVICES=1,2,0 python combined_test.py 2 $i origin
CUDA_VISIBLE_DEVICES=1,2,0 python combined_test.py 2 $i combined
done