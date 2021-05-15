class Config(object):
    train_path = '../UnsupervisedLearning-JigsawPuzzle/dataset/train'
    test_path = '../UnsupervisedLearning-JigsawPuzzle/dataset/test'

    train_csv = '../UnsupervisedLearning-JigsawPuzzle/dataset/csv/train.csv'
    valid_csv = '../UnsupervisedLearning-JigsawPuzzle/dataset/csv/valid.csv'

    pretrain = True # set True if you want to use pretrained weight
    pretrained_path = '../UnsupervisedLearning-JigsawPuzzle/model_architecture/alexnet_jigsaw_in1k_pretext.pkl'

    label_dict = {'butterfly': 0, 'cat': 1, 'chicken': 2, 'cow': 3, 'dog': 4, 'elephant': 5, 'horse': 6, 'sheep': 7, 'spider': 8, 'squirrel': 9}

    model_name = 'Jigsaw_Alexnet.pth'
    classifier_name = 'Alexnet.pth'

    checkpoint_path = '../UnsupervisedLearning-JigsawPuzzle/save_model/' + model_name
    checkpoint_classifier_path = '../UnsupervisedLearning-JigsawPuzzle/save_model/' + classifier_name

    valid_ratio = 0.2

    train_batch = 64
    valid_batch = 64

    lr = 3e-4

    num_epochs = 300

    optimizer = "SGD"

    scheduler = 'step'
    milestones = [30, 60] #multiLR
    gamma_period = 20