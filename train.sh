python main.py kinetics RGB kinetics_files/train_all.txt  kinetics_files/val_all.txt --arch resnet101 --snapshot_pref kinetics_i3dresnet101_ \
                --lr 0.001 --lr_steps 20 80 --epochs 120 \
                -b 32 -j 8 --dropout 0.5 -p 20 --gd 20
