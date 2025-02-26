import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.accuracy import BinaryAccuracy
import math
from sklearn import datasets
import pandas as pd
from sklearn.svm import LinearSVC
import random
from scipy.stats import entropy
from scipy.spatial import distance

class Data(Dataset):
  def __init__(self, X_train, y_train):

    self.X = torch.from_numpy(X_train.astype(np.float32))

    self.y = torch.from_numpy(y_train).type(torch.LongTensor)
    self.len = self.X.shape[0]

  def __getitem__(self, index):
    return self.X[index], self.y[index]

  def __len__(self):
    return self.len


def norm(x1, x2):

    if len(x1) != len(x2):
        raise "Differently sized vectors"
    
    dists = []
    for i in range(len(x1)):
        d = distance.euclidean(x1[i],x2[i])
        dists.append(d)

    return np.average(np.array(dists))


class LayerMemory:
    def __init__(self):
        self.input = []
        self.linear_output = []
        self.activation_output = []
        self.dimensional_cut = 0.
        self.enthophy_cut = 0.
        self.prunability = 0.
        self.vertical_norm = 0.

class ElasticNet(nn.Module):
    def __init__(self, layers_composition, inputs, outputs, activation = 'relu'):

        super(ElasticNet, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers_composition = [inputs] + layers_composition + [outputs]

        for i in range(len(self.layers_composition)-1):
            self.layers.append(nn.Linear(self.layers_composition[i],self.layers_composition[i+1]))

        #self.layers.append = nn.Linear(layers_composition[-1],outputs)

        #improve this later:
        self.activation = activation
        if self.activation not in ['relu','sigmoid']:
            self.activation = 'relu'
        
        #require tracking to be set manually:
        self.track = False

        self.memory = [LayerMemory() for _ in range(len(self.layers))]

    def setTracking(self, value):
        self.track = value

    def addNeurons(self, layer: int):

        if layer == -1 or layer == len(self.layers) -1:
            raise Exception('Cannot add neurons to the output layer')

        neuron_to_copy_idx = random.randint(0,self.layers_composition[layer+1]-1)

        old_weight_in = self.layers[layer].weight.detach()
        old_weight_out = self.layers[layer+1].weight.detach()
        old_bias_in = self.layers[layer].bias.detach()
        old_bias_out = self.layers[layer+1].bias.detach()

        random_weight_split = random.random()
        self.layers_composition[layer+1] = self.layers_composition[layer+1] + 1
        self.layers[layer] = nn.Linear(self.layers_composition[layer],self.layers_composition[layer+1])
        self.layers[layer+1] = nn.Linear(self.layers_composition[layer+1], self.layers_composition[layer+2])

        old_weight_in[neuron_to_copy_idx] = old_weight_in[neuron_to_copy_idx] * (1. - random_weight_split)
        self.layers[layer].weight = nn.Parameter(torch.concat((old_weight_in,torch.unsqueeze(old_weight_in[neuron_to_copy_idx],dim=0)*random_weight_split),dim=0))
        old_bias_in[neuron_to_copy_idx] *= (1. - random_weight_split)
        old_bias_in = torch.cat((old_bias_in, torch.unsqueeze(old_bias_in[neuron_to_copy_idx] * random_weight_split, dim = -1)))
        self.layers[layer].bias = nn.Parameter(old_bias_in)

        old_weight_out[:,neuron_to_copy_idx] = old_weight_out[:,neuron_to_copy_idx] * (1. - random_weight_split)
        self.layers[layer+1].weight = nn.Parameter(torch.cat((old_weight_out,torch.unsqueeze(old_weight_out[:,neuron_to_copy_idx]*random_weight_split,dim=-1)),dim=-1))
        self.layers[layer+1].bias = nn.Parameter(old_bias_out)

        if self.layers[layer].out_features != self.layers_composition[layer+1]:
            raise Exception("Dimensions error")

        if self.layers[layer+1].in_features != self.layers_composition[layer+1]:
            raise Exception("Dimensions error")

    def addLayer(self, after_layer: int):
        size = self.layers_composition[after_layer+1]
        self.layers.insert(after_layer+1, nn.Linear(size,size))
        self.layers_composition.insert(after_layer+1, size)
        #memory:
        self.memory.insert(after_layer+1,LayerMemory())

        # ♪YOU WERE THE ONE TO CUT ME SO I'LL BLEED FOREVER♪

        noise_intensity = 0.001

        stdv = 1. / math.sqrt(size)
        weights = torch.eye(size) + torch.rand((size,size)) * stdv * noise_intensity
        biases = torch.rand((size)) * stdv * noise_intensity

        self.layers[after_layer+1].weight = nn.Parameter(weights)

        self.layers[after_layer+1].bias = nn.Parameter(biases)

        if self.layers[after_layer].weight.shape[0] != self.layers_composition[after_layer+1] or self.layers[after_layer].weight.shape[1] != self.layers_composition[after_layer]:
            raise Exception('Skićkały się wymiary')


    def removeNeurons(self, layer_idx):

        if self.layers_composition[layer_idx+1] == 1:
            raise Exception('Cannot reduce neurons in 1-neuron layer')

        old_weights_in = self.layers[layer_idx].weight
        old_weights_out = self.layers[layer_idx+1].weight
        old_bias_in = self.layers[layer_idx].bias

        norms = []

        for i in range(old_weights_in.size()[0]):
            #norm needed 
            norms.append((i,torch.norm(old_weights_in[i])))
        norms.sort(key=lambda x: x[1].item())

        neuron_idx = norms[0][0]

        if neuron_idx >= self.layers_composition[layer_idx+1]:
            raise Exception("Something went like very, very wrong with dimensions")

        print('Removing at idx: ', neuron_idx)

        #review if biasses are correct
        self.layers_composition[layer_idx+1] = self.layers_composition[layer_idx+1] - 1

        self.layers[layer_idx] = nn.Linear(self.layers_composition[layer_idx],self.layers_composition[layer_idx+1])
        self.layers[layer_idx+1] = nn.Linear(self.layers_composition[layer_idx+1],self.layers_composition[layer_idx+2])
        #gdzieś tu jest błąd
        if neuron_idx == 0:
            new_weights_in = old_weights_in[1:]
            new_weights_out = old_weights_out[:,1:]
            old_bias_in = old_bias_in[1:]
            
        elif neuron_idx >= self.layers_composition[layer_idx+1]-1:
            new_weights_in = old_weights_in[:-1]
            new_weights_out = old_weights_out[:,:-1]
            old_bias_in = old_bias_in[:-1]
        else:
            new_weights_in = torch.cat((old_weights_in[:neuron_idx],old_weights_in[neuron_idx+1:]),dim=0)
            new_weights_out = torch.cat((old_weights_out[:,:neuron_idx],old_weights_out[:,neuron_idx+1:]),dim=-1)
            old_bias_in = torch.cat((old_bias_in[:neuron_idx],old_bias_in[neuron_idx+1:]), dim = -1)
        self.layers[layer_idx].weight = nn.Parameter(new_weights_in)
        self.layers[layer_idx+1].weight = nn.Parameter(new_weights_out)


        self.layers[layer_idx].bias = nn.Parameter(old_bias_in)

        if self.layers[layer_idx].out_features != self.layers_composition[layer_idx+1]:
            raise Exception("Dimensions error")

        if self.layers[layer_idx+1].in_features != self.layers_composition[layer_idx+1]:
            raise Exception("Dimensions error")

    def removeLayer(self, layer_idx):
            
        with torch.no_grad():

            if layer_idx >= len(self.layers)-1:
                raise Exception('Cannot remove the last layer from the neural network')

            #todo shrinking algorithm

            size_difference = self.layers_composition[layer_idx+1] - self.layers_composition[layer_idx]


            if size_difference == 0:
                # simply remove
                pass


            if size_difference > 0:
                # required expansion of the first layer

                weights = self.layers[layer_idx-1].weight.detach()
                biases = self.layers[layer_idx-1].bias.detach()
                
                self.layers[layer_idx-1] = nn.Linear(self.layers_composition[layer_idx-1],self.layers_composition[layer_idx+1])
                for i in range(size_difference):
                    neuron_to_copy_idx = random.randint(0,self.layers_composition[layer_idx-1]-1)
                    random_weight_split = random.random()
                    weights[neuron_to_copy_idx] = weights[neuron_to_copy_idx] * (1. - random_weight_split)
                    weights = nn.Parameter(torch.concat((weights,torch.unsqueeze(weights[neuron_to_copy_idx],dim=0)*random_weight_split),dim=0))
                    biases[neuron_to_copy_idx] *= (1. - random_weight_split)
                    biases = torch.cat((biases, torch.unsqueeze(biases[neuron_to_copy_idx] * random_weight_split, dim = -1)))

                self.layers[layer_idx-1].weight = nn.Parameter(weights)
                self.layers[layer_idx-1].bias = nn.Parameter(biases)

                '''
                norms = []

                for i in range(old_weights.size()[-1]):
                    #norm needed 
                    norms.append((i,torch.norm(old_weights[:,i])))
                norms.sort(key=lambda x: x[1].item())

                idx_to_remove = []

                for i in range(size_difference):
                    idx_to_remove.append(norms[i][0])

                for i in idx_to_remove:
                    old_weights = torch.cat((old_weights[:,:i],old_weights[:,i+1:]),dim=-1)
                
                self.layers[layer_idx+1] = nn.Linear(self.layers_composition[layer_idx-1],self.layers_composition[layer_idx+1])
                self.layers[layer_idx+1].weight = nn.Parameter(old_weights)
            '''
            if size_difference < 0:
                # required reduction of the first layer, and it's going to be performed per norms
                # THINK ABOUT IT
                weights = self.layers[layer_idx-1].weight
                biases = self.layers[layer_idx-1].bias


                self.layers[layer_idx-1] = nn.Linear(self.layers_composition[layer_idx-1],self.layers_composition[layer_idx+1])

                for i in range(abs(size_difference)):

                    norms = []

                    for j in range(weights.size()[0]):
                        norms.append((j,torch.norm(weights[j])))
                    norms.sort(key=lambda x: x[1].item())

                    neuron_idx = norms[0][0]
                    if neuron_idx == 0:
                        weights = weights[1:]
                        biases = biases[1:]

                    elif neuron_idx >= self.layers_composition[layer_idx]-1:
                        weights = weights[:-1]
                        biases = biases[:-1]
                    else:
                        weights = torch.cat((weights[:neuron_idx-1],weights[neuron_idx+1:]),dim=0)
                        biases = torch.cat((biases[:neuron_idx-1],biases[neuron_idx+1:]), dim = -1)

            # biasy powinny się zgadzać
            del self.layers[layer_idx]
            self.layers_composition.pop(layer_idx)
            self.memory.pop(layer_idx)

            if self.layers[layer_idx-1].weight.shape[0] != self.layers_composition[layer_idx] or self.layers[layer_idx-1].weight.shape[1] != self.layers_composition[layer_idx-1]:
                raise Exception('Skićkały się wymiary')
    
    def calculate_gdv(self):
        pass


    def assign_redundance_scores(self):
        threshold = 1e-4
        cutoff = 1e-1
        for i, mem in enumerate(self.memory):
            pca = PCA()
            pca.fit(mem.linear_output)
            #this normalization may not be realy necessary
            eigenvalues = np.round(pca.explained_variance_ / np.max(pca.explained_variance_),3) if np.max(pca.explained_variance_) > threshold else np.round(pca.explained_variance_,3)

            cut_ratio = sum(1 for i in eigenvalues if i < cutoff) / len(eigenvalues)

            # "żyjemy w czasach, w których termostat ma bardziej stabilne poglądy niż właściciel" <3

            mem.dimensional_cut = cut_ratio

            # entrophy
            data_entropy = 0.
            
            #if np.sum(mem.activation_output) < 1e3:
            #    data_entropy = entropy(mem.activation_output+1e3)
            #else:
            #    data_entropy = entropy(mem.activation_output)

            #mem.enthophy_cut = data_entropy

            # prunability
            if i < len(self.memory) - 1:
                weights = self.layers[i].weight.detach().numpy()

            pruning_threshold = 1e-2 * np.max(weights)
            prunable_weights = np.count_nonzero(np.absolute(weights) < pruning_threshold)
            mem.prunability = prunable_weights / weights.size

    def assign_redundance_vertical(self):
        for i, mem in enumerate(self.memory):
            if mem.input.shape[-1] == mem.activation_output.shape[-1]:
                difference = norm(mem.input,mem.activation_output)
                mem.vertical_norm = difference / math.sqrt(mem.input.shape[-1])   
            else:
                # try to pca the larger dimension, if it happens PCA yields some nonsense results it's even better
                if mem.input.shape[-1] > mem.activation_output.shape[-1]:
                    pca = PCA(n_components=mem.activation_output.shape[-1])
                    reduced_data = pca.fit_transform(mem.input)
                    difference = norm(reduced_data,mem.activation_output)
                    mem.vertical_norm = difference / math.sqrt(reduced_data.shape[-1])
                else:
                    pca = PCA(n_components=mem.input.shape[-1])
                    reduced_data = pca.fit_transform(mem.activation_output)
                    difference = norm(mem.input,reduced_data)
                    mem.vertical_norm = difference / math.sqrt(reduced_data.shape[-1])


    def get_aggregated_memory_arrays(self):
        dimensional_cuts = []
        entrophy_cuts = []
        prunability_cuts = []
        norms = []

        for mem in self.memory:
            dimensional_cuts.append(mem.dimensional_cut)
            entrophy_cuts.append(mem.enthophy_cut)
            prunability_cuts.append(mem.prunability)
            norms.append(mem.vertical_norm)

        return dimensional_cuts, entrophy_cuts, prunability_cuts, norms
    

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.track:
                self.memory[i].input = x.detach().numpy()
            x = layer(x)
            if self.track:
                #self.memory[i].linear_output.append(x.detach().numpy())
                self.memory[i].linear_output = x.detach().numpy()

            if i != len(self.layers):
                # TODO: clean this
                if self.activation == 'relu':
                    x = torch.relu(x)
                elif self.activation == 'sigmoid':
                    x = torch.sigmoid(x)
                else:
                    x = torch.relu(x)
            if self.track:
                self.memory[i].activation_output = x.detach().numpy()
        return x
    
    