from abc import ABCMeta, abstractmethod
from utils import compute_return_tsp
import torch

class Inference(metaclass=ABCMeta):
    @abstractmethod
    def infer(self, tsp_model, instances):
        raise NotImplementedError

class InferenceSampling(Inference):

    def __init__(self, device, nb_samples=100):
        self.device = device
        self.nb_samples = nb_samples

    def infer(self, model, instances):
        model.eval()
        model.set_decode_mode('sample')


        solutions, tour_lengths = None, None
        with torch.no_grad():
            for _ in range(self.nb_samples):
                _, solutions_tmp = model(instances)
                tour_lengths_tmp = compute_return_tsp(instances, solutions_tmp, self.device)

                if solutions is None:
                    solutions, tour_lengths = solutions_tmp, tour_lengths_tmp
                else:
                    for i in range(tour_lengths.size()[0]):
                        if tour_lengths_tmp[i].item() < tour_lengths[i].item():
                            tour_lengths[i] = tour_lengths_tmp[i]
                            solutions[i] = solutions_tmp[i]

        return solutions, tour_lengths

class InferenceGreedy(Inference):

    def __init__(self, device):
        self.device = device

    def infer(self, model, instances):
        model.eval()
        model.set_decode_mode('greedy')
        with torch.no_grad():
            _, solutions= model(instances)
            tour_length = compute_return_tsp(instances, solutions, self.device)
        return solutions, tour_length
