# & means run these methods at the same time, and && means run these methods one by one
python -u test_3D.py --root_path ../data/BraTS2019 --exp BraTS2019/Fully_supervised_25 --model unet_3D &&
python -u test_3D.py --root_path ../data/BraTS2019 --exp BraTS2019/Fully_supervised_250 --model unet_3D && 
python -u test_3D.py --root_path ../data/BraTS2019 --exp BraTS2019/Mean_Teacher_25 --model unet_3D && 
python -u test_3D.py --root_path ../data/BraTS2019 --exp BraTS2019/Uncertainty_Aware_Mean_Teacher_25 --model unet_3D && 
python -u test_3D.py --root_path ../data/BraTS2019 --exp BraTS2019/Interpolation_Consistency_Training_25 --model unet_3D && 
python -u test_3D.py --root_path ../data/BraTS2019 --exp BraTS2019/Entropy_Minimization_25 --model unet_3D && 
python -u test_3D.py --root_path ../data/BraTS2019 --exp BraTS2019/Cross_Pseudo_Supervision_25 --model unet_3D && 
python -u test_3D.py --root_path ../data/BraTS2019 --exp BraTS2019/Adversarial_Network_25 --model unet_3D && 
python -u test_3D.py --root_path ../data/BraTS2019 --exp BraTS2019/Uncertainty_Rectified_Pyramid_Consistency_25 --model unet_3D_dv_semi