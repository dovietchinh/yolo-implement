import torch 
import numpy as np
import logging 

LOGGER = logging.getlogger(__name__)

class YoloLoss(torch.nn.Module):

    def __init__(self,*kwargs):
        super(YoloLoss,self).__init__()
        self.coefficient = kwargs.get('coefficient')
        self.shape = 7,7,30
    
    def SSE(self,x,y):
        return (torch.sqrt(x)-torch.sqrt(y))**2
    
    def forward(self,predict,ground_truth): 
        """[summary]

        Args:
            predict (torch.Tensor): N*7*7*30
            ground_truth (torch.Tensor): N*7*7*30
        """
        predict = torch.view(predict,(predict.shape[0],-1))
        ground_truth = torch.view(ground_truth,(ground_truth.shape[0],-1))
        cordinate_loss = 0
        shape_loss = 0
        class_loss = 0
        for i in range(0,7):
            for j in range(0,7):
                cordinate_loss += ground_truth[i,j,24] * ((ground_truth[20]-predict[20])**2 + (ground_truth[21]-predict[21])**2) +\
                                ground_truth[i,j,29] * ((ground_truth[25]-predict[25])**2 + (ground_truth[26]-predict[26])**2)
                shape_loss += ground_truth[i,j,24] * ( SSE(ground_truth[22],predict[22])+ SSE(ground_truth[23],predict[23])) +\
                        ground_truth[i,j,29] * ( SSE(ground_truth[27],predict[27])+ SSE(ground_truth[28],predict[28]))
                for k in range(20):
                    class_loss +=  ground_truth[i,j,24] ()      


                


