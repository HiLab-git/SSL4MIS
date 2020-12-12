CUDA_VISIBLE_DEVICES=0 python train_fully_supervised_2D.py --root_path ../data/ACDC --exp ACDC/Fully_Supervised --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=0 python train_entropy_minimization_2D.py --root_path ../data/ACDC --exp ACDC/Entropy_Minimization --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=0 python train_interpolation_consistency_training_2D.py --root_path ../data/ACDC --exp ACDC/Interpolation_Consistency_Training --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=0 python train_mean_teacher_2D.py --root_path ../data/ACDC --exp ACDC/Mean_Teacher --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=0 python train_uncertainty_aware_mean_teacher_2D.py --root_path ../data/ACDC --exp ACDC/Uncertainty_Aware_Mean_Teacher --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=0 python train_adversarial_network_2D.py --root_path ../data/ACDC --exp ACDC/Adversarial_Network --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=0 python train_uncertainty_rectified_pyramid_consistency_2D.py --root_path ../data/ACDC --exp ACDC/Uncertainty_Rectified_Pyramid_Consistency --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=0 python train_fully_supervised_2D.py --root_path ../data/ACDC --exp ACDC/Fully_Supervised --num_classes 4 --labeled_num 140