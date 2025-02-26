import elastictransformer as et
import torch
from torch import nn
import math
import matplotlib.pyplot as plt
import pickle
import os

class ControlNeuron:
    pass

class MRTM(nn.Module):
    def __init__(self, t: et.ElasticTransformer, dump_name = "", dump_folder = "./"):
        super().__init__()
        
        self.t = t

        self.dump_name = dump_name
        self.dump_folder = dump_folder

        self.controlNeurons = {
            "evolution_factor": ControlNeuron(),
            "redundancy_horizontal_expand": ControlNeuron(),
            "redundancy_horizontal_shrink": ControlNeuron(),
            "redundancy_vertical_expand": ControlNeuron(),
            "redundancy_vertical_shrink": ControlNeuron(),
            "bypass": ControlNeuron(),
        }

        self.losses = []

    
    def run(self, total_epochs, increment, device_id, train_dataset, vertical_control_delay, flow_control_dataset,
            target_dim_threshold,target_dim_threshold_upper,target_norm_threshold,target_norm_threshold_upper,
            head_limit, stack_limit, backup = True, backup_delay = 50, lr = 0.001):

        loss_history = []

        epochs = 0

        losses = []
        r_square = []
        mean_prunabilities = []
        mean_dims = []


        optimizer = torch.optim.Adam(self.t.parameters(),lr=lr)
        
        while epochs < total_epochs:



            self.t.setTracking(False)
            _, losses = et.train_cuda(self.t, train_dataset, device_id, epochs = increment, verbose_delay=-1, optimizer=optimizer)
            print('Epoch: ', epochs, ' running loss: ', losses[-1])
            loss_history.append(losses[-1])
            epochs +=1
            
            self.t.clearMemory()

            with torch.no_grad():
                self.t.setTracking(True)
                validation = et.validate_cuda(self.t,flow_control_dataset,device_id,batch_size=1440)

                self.t.assignHorizontalRedundanceScores()
                self.t.assignVerticalRedundanceScores()
                print('Scores assigned')
                self.t.setTracking(False)



            print('hris (encoders)')
            for e in self.t.encoder_stack.encoders:
                #hri = e.mha.output_memory.dimenisonal_reductability
                hri = math.sqrt(e.mha.output_memory.dimenisonal_reductability*e.mha.output_memory.pruningability)
                print(hri)
                if hri < target_dim_threshold and len(e.mha.heads) < head_limit:
                    e.mha.add_head_horizontal()
                if hri > target_dim_threshold_upper:
                    if len(e.mha.heads) > 1:
                        e.mha.remove_head_horizontal()
            
            print('hris (decoders)')
            for d in self.t.decoder_stack.decoders:
                #hri = d.mha.output_memory.dimenisonal_reductability
                hri = math.sqrt(d.self_mha.output_memory.dimenisonal_reductability*d.self_mha.output_memory.pruningability)
                print(hri)
                if hri < target_dim_threshold and len(d.self_mha.heads) < head_limit :
                    d.self_mha.add_head_horizontal()
                    print('adding a head')
                if hri > target_dim_threshold_upper:
                    if len(d.self_mha.heads) > 1:
                        d.self_mha.remove_head_horizontal()
                        print('removing a head')

                hri = math.sqrt(d.ed_mha.output_memory.dimenisonal_reductability*d.ed_mha.output_memory.pruningability)
                print(hri)
                if hri < target_dim_threshold and len(d.ed_mha.heads) < head_limit:
                    d.ed_mha.add_head_horizontal()
                    print('adding a head')
                if hri > target_dim_threshold_upper:
                    if len(d.ed_mha.heads) > 1:
                        d.ed_mha.remove_head_horizontal()
                        print('removing a head')

            print('heads:')
            print('encoders:')
            for e in self.t.encoder_stack.encoders:
                print(len(e.mha.heads))
            print('decoders:')
            for d in self.t.decoder_stack.decoders:
                print(len(d.self_mha.heads), ',', len(d.ed_mha.heads))

            sum_of_heads = 0
            sum_of_mhas = 0
            for e in self.t.encoder_stack.encoders:
                sum_of_heads += len(e.mha.heads)
            sum_of_mhas += len(self.t.encoder_stack.encoders)
            for d in self.t.decoder_stack.decoders:
                sum_of_heads += len(d.ed_mha.heads)
                sum_of_heads += len(d.self_mha.heads)
            sum_of_mhas += 2*len(self.t.decoder_stack.decoders)

            # update config to match the average head number:
            new_head_number = int(sum_of_heads/sum_of_mhas)

            if new_head_number > 0:
                self.t.config.dictionary['n_heads'] = new_head_number


            if epochs % vertical_control_delay == 0:
                self.t.assignVerticalRedundanceScores()

                print('vris')
                for i, e in enumerate(self.t.encoder_stack.encoders):
                    vri = e.vertical_redundance
                    print(vri)
                    if vri > target_norm_threshold_upper:
                        head_number = len(e.mha.heads)
                        #self.t.add_encoder(i,head_number=head_number)
                        self.t.addEncoder(i)
                        #self.t.encoder_stack.encoders.insert(i, et.EncoderLayer(self.t.config,randomize = True,head_number=head_number))
                        print('Adding an encoder')
                        break
                    else:
                        if vri < target_norm_threshold:
                            if len(self.t.encoder_stack.encoders) > 1:
                                del self.t.encoder_stack.encoders[i]
                                print('Deleting an encoder')
                                break

                for i, d in enumerate(self.t.decoder_stack.decoders):
                    print(d.vertical_redundance)
                    if d.vertical_redundance > target_norm_threshold_upper:
                        #self.t.add_decoder(i,head_number=head_number)
                        self.t.addDecoder(i)
                        print('Adding a decoder')
                        break
                    else:
                        if d.vertical_redundance < target_norm_threshold:
                            if len(self.t.decoder_stack.decoders) > 1:
                                del self.t.decoder_stack.decoders[i]
                                print('Deleting a decoder')
                                break


            # we need to repeat this in case network has changed
            optimizer = torch.optim.Adam(self.t.parameters())

            if epochs > 0 and epochs % backup_delay == 0 and backup == True:
                pickle.dump()

        self.losses = losses

    '''
    def generate_report(self, path = "./" name = ""):
        directory_created = False
        try:
            os.mkdir(path)
            directory_created = True
        except FileExistsError:
            print(f"One or more directories in '{path}' already exist.")
            directory_created = True
        except PermissionError:
            print(f"Permission denied: Unable to create '{path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        if directory_created:
            
    ''' 
    
    def print_structure(self):
        print('Structure')
        print('Encoder stack: ')
        for e in self.t.encoder_stack.encoders:
            print(len(e.mha.heads))

        print('Decoder stack: ')
        for d in self.t.decoder_stack.decoders:
            print(len(d.self_mha.heads), ' ', len(d.ed_mha.heads))

    def print_loss(self):
        plt.plot(self.losses)
        print(self.losses)
        plt.show()
