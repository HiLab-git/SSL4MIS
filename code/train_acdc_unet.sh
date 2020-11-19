CUDA_VISIBLE_DEVICES=2 python train_unet_2D_fully_supervised.py --root_path ../data/ACDC --exp ACDC/Fully_Supervised --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=2 python train_unet_2D_dv_fully_supervised.py --root_path ../data/ACDC --exp ACDC/Fully_Supervised_DV --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=2 python train_unet_2D_entropy_minimization.py --root_path ../data/ACDC --exp ACDC/Entropy_Minimization --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=2 python train_unet_2D_interpolation_consistency_training.py --root_path ../data/ACDC --exp ACDC/Interpolation_Consistency_Training --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=2 python train_unet_2D_mean_teacher.py --root_path ../data/ACDC --exp ACDC/Mean_Teacher --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=2 python train_unet_2D_uncertainty_aware_mean_teacher.py --root_path ../data/ACDC --exp ACDC/Uncertainty_Aware_Mean_Teacher --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=2 python train_unet_2D_adversarial_network.py --root_path ../data/ACDC --exp ACDC/Adversarial_Network --num_classes 4 --labeled_num 7