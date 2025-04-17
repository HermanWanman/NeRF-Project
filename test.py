import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128): # embedding_dim_pos (= feature dimension of the position encoding) and 
        super(NerfModel, self).__init__()                                                # embedding_dim_direction are positional data needed for this NeRF implementation 

        # Block1 and block2 calculate the density (sigma) or "opaque-ness" of the "volumetric representation integral" and this is not dependent on the direction
        # that is why the direction is first used in block3 since that is where it is first needed. No need to confuse the network with data that is not important
        # this entire sequence can be seen in Fig 7 of the paper "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"

        self.block1 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), nn.ReLU(), # simple MLP (= Multi-Layer-Perceptron) with ReLU-activation
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        
        self.block2 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim), nn.ReLU(), # we add the hidden_dim and add it into this block since we also want the output from the first block
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),                             # block2 does the same as block1 but also takes the output of block1 as an input
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1), )                                  # here we have an output-dimension of hidden_dim+1 because that last output is the density that the network has 
                                                                                                              # calculated. There is also no activation for this last layer
        
        # The color is dependent on the direction and thus we input it into block3 along with the density (calculated by block1 and block2) which is also needed for the RGB-values
        
        self.block3 = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2), nn.ReLU(), ) # here we start calculating the color to get the RGB-values to output
        self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(), )  # here we reduce the dimension of the outputs to create the RGB-values to output

        # now we save the values for the direction and the position 

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod   # This function is made because neural networks makes it so that high frequency of color and geometry changes tends to be lower than desired if
                    # the input is only our 3d-positioning and 2d-viewing direction. To combat this we, somewhat artificially, increase the dimension with every value in the position and direction
                    # through the formula sin(2^(j)*pi*p), cos(2^(j)*pi*p) where j = {0,1,2, ... L-1} and L = "half of the dimensional space we map the inputs to"
                    # This function maps 3d-input to a 63d-feature vector and 2d-input to a 24d-feature vector
                    # Seen in section 5.1 in the paper
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)
    
    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos) # we embedd the position which means we transform it as described above
        emb_d = self.positional_encoding(d, self.embedding_dim_direction) # we embedd the direction which means we transform it as described above
        h = self.block1(emb_x) # compute the first part of the desity calculation
        tmp = self.block2(torch.cat((h,emb_x), dim=1))  # this line calculates the second part of the density calculation on the concatenated 
                                                        # (stacked horizontally aka values on row 1 in h and the values on row 1 in emb_x are all in row 1 in the concatenated matrix) values of the 
                                                        # last post-activation values from block1 and the encoded positional input

        h,sigma = tmp[:,:-1], self.relu(tmp[:,-1])      # takes the reversed preactivations of the block2-output which represents the density and inputs it into h for the next block
                                                        # it also activates the density and saves it into the sigma-variable
        h = self.block3(torch.cat((h, emb_d), dim=1))   # uses the preactivations from the line above with the addition of the direction to run the next part of the nn
        c = self.block4(h)                              # converts the new h-values into the 3d RGB-value
        return c, sigma                                 # returns the RGB-color-value and the density
    

def compute_accumulated_transmittance(alphas):  # implementing the part of formula (3) on page 6 in the paper that calculates the T_i-value or the accumulated transmittance
                                                # Formula (3) calculates the ray traced through each pixel for our view (or pinhole camera) 
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device), 
                        accumulated_transmittance[:, :-1]), dim=1)


# Here we start making the rendering of the program

def render_rays(nerf_model, ray_origins, ray_directions,hn=0, hf=0.5, nb_bins=192): #nb_bins = the number of sums used to approximate the integration
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins) # we sample t between the points hn and hf to be used for getting x-values along the ray
    mid = (t[:,:-1] + t[:,1:])/ 2. # don't understand
    lower = torch.cat((t[:,:1], mid), -1) # don't understand
    upper = torch.cat((mid, t[:,-1:]), -1) # don't understand
    u = torch.rand(t.shape, device=device) # calculate u-value for raytracing
    t = lower + (upper - lower) * u # size [batch_size, nb_bins] standard t-value for ray tracing

    delta = torch.cat((t[:,1:] - t[:,:-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1) # calculates the width of each bin (or part of the sum to approximate the integral)
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1) # the position of a ray at time t is the origin + t*direction
                                                                                # size [batch_size, nb_bins, 3]
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)
    
    colors,sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3)) # we retrieve a color and a density for each point x along the ray
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    # now we use these calculated values to compute the color of each ray
    alpha = 1 - torch.exp(-sigma*delta) # size [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1-alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1) # pixel values
    weight_sum = weights.sum(-1).sum(-1) # regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1) # + 1 - weight_sum.unsqueeze(-1) assumes we have a white background and makes it so

# Here we make our training function that does supervised learning by using the 2d pictures in the training data to know what color the ray is supposed to be in order to train the model
def train(nerf_model, optimizer, scheduler,testing_dataset, data_loader, device='cuda', hn=0, hf=1, nb_epochs=int(1e5), nb_bins=192, H=400, W=400):
    training_loss = []
    for _ in tqdm(range(nb_epochs)): #iterate over the epochs
        for batch in data_loader: # iterate over the data in our dataloader
            ray_origins = batch[:,:3].to(device) # fetch the origin from the known data
            ray_directions = batch[:, 3:6].to(device) # fetch the direction of the known data
            ground_truth_px_values = batch[:, 6:].to(device) # fetch the actual value that is supposed to be at this pixel

            generated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
            loss = ((ground_truth_px_values - generated_px_values) ** 2).sum() # compare the values with MSE

            # We use the optimizer to do gradient descent

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
        scheduler.step() # this mechanism is used to refine details at the end of training since NeRF usually converges fast to the shape but the detalis are helped by not letting it converge to early which is done by the scheduler


        # after each epoch we test the model but since testing takes a while with NeRF we don't test the whole dataset we only test a few values

        for img_index in range(200):
            test(hn, hf, testing_dataset, img_index=img_index, nb_bins=nb_bins, H=H, W=W)
    return training_loss

@torch.no_grad()
def test(hn, hf, dataset, chunck_size=10, img_index=0, nb_bins=192, H=400, W=400, device='cuda'):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index *H * W: (img_index + 1) * H * W, 3:6]

    data = []
    for i in range(int(np.ceil(H / chunck_size))): # we divide the image into batches to prevent out-of-memory issues
        ray_origins_ = ray_origins[i * W * chunck_size: (i + 1) * W * chunck_size].to(device)
        ray_directions_ = ray_directions[i * W * chunck_size: (i + 1) * W * chunck_size].to(device)
        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)

    # show our generated image using matplotlib

    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)
    plt.figure()
    plt.imshow(img)
    plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    device = 'cuda'
    training_dataset = torch.from_numpy(np.load('training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))
    model = NerfModel(hidden_dim=256).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4) # Adam is an optimization algorithm that can be used instead of SGD for deep learning models
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones= [2, 4, 8], gamma=0.5) 

    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True) 
    train(model, model_optimizer, scheduler,testing_dataset, data_loader, nb_epochs=16, device=device, hn=2, hf=6, nb_bins=192, H=400, W=400)
    
