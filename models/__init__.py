from .Fast_ACV import Fast_ACVNet
from .Fast_ACV_plus import Fast_ACVNet_plus
from .loss import model_loss_train, model_loss_test

__models__ = {
    "Fast_ACVNet": Fast_ACVNet, 
    "Fast_ACVNet_plus": Fast_ACVNet_plus
}
