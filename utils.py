from torch.nn import L1Loss


class L1Evaluator:
    """
    L1 Evaluator similar to OGB evaluators.
    """
    def __init__(self):
        self.evaluator = L1Loss()

    def eval(self, x):
        return self.evaluator(x['y_pred'], x['y_true'])
