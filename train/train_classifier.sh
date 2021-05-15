pretrained_path = '../UnsupervisedLearning-JigsawPuzzle/save_model/Jigsaw_Alexnet.pth'
checkpoint_path = '../UnsupervisedLearning-JigsawPuzzle/save_model/Alexnet.pth'
python train_classifier.py --pretrained ${pretrained_path} --checkpoint ${checkpoint_path} 