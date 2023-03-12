import torch
import torch.nn as nn

class Emitter(nn.Module):
    #Parameterizes `p(x_t | z_t)`

    def __init__(self, input_dim, latent_dim = 3, nn_dim = 32, nn_layers = 2):
        super(Emitter, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.nn_dim = nn_dim
        self.nn_layers = nn_layers

        if nn_layers == 3:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, nn_dim),
                nn.ReLU(),
                nn.Linear(nn_dim, nn_dim),
                nn.ReLU(),
                nn.Linear(nn_dim, input_dim))
        elif nn_layers == 2:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, nn_dim),
                nn.Tanh(),
                nn.Linear(nn_dim, input_dim))
        elif nn_layers == 1:
            # ## Incremental
            # self.bias = nn.Parameter(torch.zeros(input_dim))
            # self.weights = nn.Parameter(torch.zeros(latent_dim, input_dim))

            # ## General
            # self.decoder = nn.Sequential(nn.Linear(latent_dim, input_dim))

            # ## Polynomial
            # poly_order = 10
            # self.w = nn.Parameter(torch.zeros( latent_dim+1,poly_order+1,))
            # nn.init.uniform_(self.w)

            # ## Radial basis
            # num_of_basis = 10
            # mu = torch.tensor(range(num_of_basis))/num_of_basis
            # self.mu = nn.Parameter(mu.unsqueeze(0).repeat(latent_dim+1,1))
            # self.l = torch.ones(latent_dim+1,num_of_basis)*10.0
            # self.w = nn.Parameter(torch.empty(latent_dim+1,num_of_basis))
            # nn.init.uniform_(self.w)

            ## Radial basis with bias
            num_of_basis = 5
            mu = torch.tensor(range(num_of_basis))/num_of_basis
            self.mu = nn.Parameter(mu.unsqueeze(0).repeat(latent_dim,1))
            self.w = nn.Parameter(torch.empty(latent_dim,num_of_basis))
            nn.init.uniform_(self.w)
            self.l = torch.ones(latent_dim,num_of_basis)*10.0
            self.bias = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        # ## Incremental
        # w = torch.cumsum(self.weights,1)
        # b = torch.cumsum(self.bias,0)
        # a = b + torch.matmul(x,w)
        # return torch.exp(a)

        # ## General
        # return torch.exp(self.decoder(x))

        # ## Polynomial
        # ages = torch.tensor(range(self.input_dim))/100.0
        # a = torch.stack([torch.ones(ages.size()),ages,torch.pow(ages,2),torch.pow(ages,3),torch.pow(ages,4),torch.pow(ages,5),torch.pow(ages,6),torch.pow(ages,7),torch.pow(ages,8),torch.pow(ages,9),torch.pow(ages,10)])
        # b = self.w@a
        # x = torch.cat((torch.tensor([1]),x),0)
        # return torch.exp(x@b)

        # ## Radial basis
        # ages = torch.tensor(range(self.input_dim)).reshape(self.input_dim,1,1)/self.input_dim
        # p = torch.exp(-torch.pow((ages-self.mu)*self.l,2))
        # x = torch.cat((torch.tensor([1]),x),0)
        # return torch.exp(torch.sum(p*self.w,2) @ x)

        ## Radial basis with bias
        ages = torch.tensor(range(self.input_dim)).reshape(self.input_dim,1,1)/self.input_dim
        p = torch.exp(-torch.pow((ages-self.mu)*self.l,2))
        return torch.exp(torch.sum(p*self.w,2) @ x + self.bias)
