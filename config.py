class Config(object):
    train_path = '../UnsupervisedLearning-JigsawPuzzle/dataset/train'
    test_path = '../UnsupervisedLearning-JigsawPuzzle/dataset/test'
    pretrain = False # set True if you want to use pretrained weight
    pretrained_path = '../'

    model_name = 'Jigsaw_Alexnet.pth'

    checkpoint_path = '../UnsupervisedLearning-JigsawPuzzle/save_model/' + model_name

    valid_ratio = 0.2

    train_batch = 16
    valid_batch = 8

    lr = 2e-4

    num_epochs = 20