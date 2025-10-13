import traceback

from src.StaticClassifierTrainer import StaticClassifierTrainer


class Init:

    imgTrain = None

    def __init__(self):
        print('Init init')

        self.imgTrain = StaticClassifierTrainer(
            'ball_model'
            ,'models/'
            ,'resources/trainData/static'
            ,'resources/eval/static'
            ,2
        )
        self.imgTrain.run_train_process()




