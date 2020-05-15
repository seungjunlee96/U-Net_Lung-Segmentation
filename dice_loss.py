import torch

class DiceCoefficient(torch.autograd.Function):
    """
    Dice coefficient for individual examples
        Dice coefficient = 2 * |X n Y| / (|X| + |Y|)
                         = 1 / ( 1/Precision + 1/Recall)
    """
    def forward(self, input , target):
        self.save_for_backward(input, target)
        eps = 1e-10
        self.inter = torch.dot(input.view(-1), target.view(-1)) # inter = |X n Y|
        self.union = torch.sum(input) + torch.sum(target) # union =|X| + |Y|

        dice = 2 * self.inter.float() / (self.union.float() + eps)
        return dice

    def backward(self,grad_output):

        input,target = self.saved_variables
        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) / ( self.union * self.union )

        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def dice_coefficient(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i , c in enumerate(zip(input, target)):
        s += DiceCoefficient().forward(c[0],c[1])

    n_data = len(input)
    return s / n_data