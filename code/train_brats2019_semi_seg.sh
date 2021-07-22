# & means run these methods at the same time, and && means run these methods one by one
python -u train_fully_supervised_3D.py --labeled_num 25 --root_path ../data/BraTS2019 --max_iterations 30000 --exp BraTS2019/Fully_supervised --base_lr 0.1 &&
python -u train_fully_supervised_3D.py --labeled_num 250 --root_path ../data/BraTS2019 --max_iterations 30000 --exp BraTS2019/Fully_supervised --base_lr 0.1 &&
python -u train_adversarial_network_3D.py --labeled_num 25 --total_num 250 --root_path ../data/BraTS2019 --max_iterations 30000 --exp BraTS2019/Adversarial_Network --base_lr 0.1 &&
python -u train_entropy_minimization_3D.py --labeled_num 25 --total_num 250 --root_path ../data/BraTS2019 --max_iterations 30000 --exp BraTS2019/Entropy_Minimization --base_lr 0.1 &&
python -u train_interpolation_consistency_training_3D.py --labeled_num 25 --total_num 250 --root_path ../data/BraTS2019 --max_iterations 30000 --base_lr 0.1 --exp BraTS2019/Interpolation_Consistency_Training &&
python -u train_mean_teacher_3D.py --labeled_num 25 --total_num 250 --root_path ../data/BraTS2019 --max_iterations 30000 --exp BraTS2019/Mean_Teacher --base_lr 0.1 &&
python -u train_uncertainty_aware_mean_teacher_3D.py --labeled_num 25 --total_num 250 --root_path ../data/BraTS2019 --max_iterations 30000 --base_lr 0.1 --exp BraTS2019/Uncertainty_Aware_Mean_Teacher &&
python -u train_uncertainty_rectified_pyramid_consistency_3D.py --labeled_num 25 --total_num 250 --root_path ../data/BraTS2019 --max_iterations 30000 --base_lr 0.1 --exp BraTS2019/Uncertainty_Rectified_Pyramid_Consistency &&
python -u train_cross_pseudo_supervision_3D.py --labeled_num 25 --total_num 250 --root_path ../data/BraTS2019 --max_iterations 30000 --base_lr 0.1 --exp BraTS2019/Cross_Pseudo_Supervision