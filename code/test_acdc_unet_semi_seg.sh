CUDA_VISIBLE_DEVICES=0 python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/Fully_Supervised --num_classes 4 --labeled_num 7 && \ 
CUDA_VISIBLE_DEVICES=0 python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/Entropy_Minimization --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=0 python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/Interpolation_Consistency_Training --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=0 python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/Mean_Teacher --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=0 python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/Uncertainty_Aware_Mean_Teacher --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=0 python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/Adversarial_Network --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=0 python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/Uncertainty_Rectified_Pyramid_Consistency --model unet_urpc --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=0 python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/Fully_Supervised --num_classes 4 --labeled_num 140