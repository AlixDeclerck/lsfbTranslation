import torch.nn as nn
import torch

class SimpleLossCompute:
    """
    A simple loss compute and train function.
    (from Annotated Transformer)
    Contiguous means : Returns a contiguous in memory tensor containing the same data as self tensor.
    https://pytorch.org/docs/master/tensors.html
    """

    def __init__(self, generator, criterion, config):
        """
        :param generator: model generator
        :param criterion: Kullback-Leibler's criterion
        """
        self.KLdl = config["hyper_parameters"]["KL_divergence_loss"]
        self.using_gpu = config["learning_config"]["using_gpu"]
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        """
        :param x: model.forward result
        :param y: target
        :param norm: number of tokens
        :return: loss, loss_node
        """

        if self.KLdl:
            x = self.generator(x)
            simple_loss = (
                    self.criterion(
                        x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
                    ) / norm)

            return simple_loss.data * norm, simple_loss

        else:
            # doc : https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html#torch.nn.CosineSimilarity
            if self.using_gpu:
                cuda_value = "cuda"
            else:
                cuda_value = "cpu"

            x = self.generator(x)
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            predict = torch.squeeze(torch.matmul(x.contiguous().view(-1, x.size(-1)), torch.ones(x.size(2), 1).cuda(torch.device(cuda_value, 0))), 1)
            target = y.contiguous().view(-1)

            similarity_loss = cos(predict, target)
            return similarity_loss.data, similarity_loss
