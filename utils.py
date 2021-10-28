import torch
from math import sqrt
#computer the reward defined as the tour length
def compute_return_tsp(instance_data, route, device):
    """

    :param instance_data: [batch_size, num_cities, 2], of (x,y) coordinates
    :param route: [batch_size, num_cites], the order in which cities are visites
    :return: [batch_size, 1] : the length of the tour constructed by route
    """
    batch_size = instance_data.size(0)

    number_of_cities = route.size(1)
    length = torch.FloatTensor(torch.zeros(batch_size,1)).to(device)

    selection = instance_data[[i for i in range(batch_size)], route[:,0], :]

    for k in range(1, number_of_cities):
        next_selection =  instance_data[[i for i in range(batch_size)], route[:,k], :]
        length += torch.norm(selection - next_selection, dim=1).view(-1,1)

        selection = next_selection


    # back to the first city
    next_selection = instance_data[[i for i in range(batch_size)], route[:, 0].view(1, -1).squeeze(0).data, :]


    length += torch.norm(selection - next_selection, dim=1).view(-1, 1)

    # [batch_size]
    return length.view(-1)


def compute_return_vrp(instance_data, route, device):
    """

    :param instance_data: [batch_size, num_cities, 2], of (x,y) coordinates
    :param route: [batch_size, sol_length], the order in which cities are visites in form of [0, route1, 0, route2, 0, route3, 0, ...,0, route_k, 0]
    :return: [batch_size, 1] : the length of the tour constructed by route
    """
    batch_size = instance_data.size(0)

    number_of_cities = route.size(1)
    length = torch.FloatTensor(torch.zeros(batch_size,1)).to(device)

    selection = instance_data[[i for i in range(batch_size)], route[:,0], :]

    for k in range(1, number_of_cities):
        next_selection =  instance_data[[i for i in range(batch_size)], route[:,k], :]
        length += torch.norm(selection - next_selection, dim=1).view(-1,1)

        selection = next_selection

    # [batch_size]
    return length.view(-1)





def init_parameters(model : torch.nn.Module):
    for name, param in model.named_parameters():
        stdv = 1. / sqrt(param.size(-1))
        param.data.uniform_(-stdv, stdv)


def freeze_parameters(model : torch.nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = False