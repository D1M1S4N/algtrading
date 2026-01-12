import time
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

def generate_square_subsequent_mask(dim1: int, dim2: int):
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

def run_encoder_decoder_inference(
    model: nn.Module,
    src: torch.Tensor,
    tgt: torch.Tensor,
    forecast_window: int,
    batch_size: int,
    batch_first: bool=False
    ):

    """
    NB! This function is currently only tested on models that work with
    batch_first = False

    This function is for encoder-decoder type models in which the decoder requires
    an input, tgt, which - during training - is the target sequence. During inference,
    the values of tgt are unknown, and the values therefore have to be generated
    iteratively.

    This function returns a prediction of length forecast_window for each batch in src

    NB! If you want the inference to be done without gradient calculation,
    make sure to call this function inside the context manager torch.no_grad like:
    with torch.no_grad:
        run_encoder_decoder_inference()

    The context manager is intentionally not called inside this function to make
    it usable in cases where the function is used to compute loss that must be
    backpropagated during training and gradient calculation hence is required.

    If use_predicted_tgt = True:
    To begin with, tgt is equal to the last value of src. Then, the last element
    in the model's prediction is iteratively concatenated with tgt, such that
    at each step in the for-loop, tgt's size increases by 1. Finally, tgt will
    have the correct length (target sequence length) and the final prediction
    will be produced and returned.

    Args:
        model: An encoder-decoder type model where the decoder requires
               target values as input. Should be set to evaluation mode before
               passed to this function.

        src: The input to the model

        forecast_horizon: The desired length of the model's output, e.g. 58 if you
                         want to predict the next 58 hours of FCR prices.

        batch_size: batch size

        batch_first: If true, the shape of the model input should be
                     [batch size, input sequence length, number of features].
                     If false, [input sequence length, batch size, number of features]

    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dimension of a batched model input that contains the target sequence values
    target_seq_dim = 0 if batch_first == False else 1

    if tgt == None:

      # Take the last value of thetarget variable in all batches in src and make it tgt
      # as per the Influenza paper

      tgt = src[-1, :, :] if batch_first == False else src[:, -1, 0] # shape [1, batch_size, 1]

    else:
      tgt = tgt.to(device)

    if src.shape[2] != tgt.shape[1]:
      fill = torch.zeros(tgt.shape[0],src.shape[2]-tgt.shape[1]).cuda()
      tgt = torch.cat((tgt,fill),dim=1)

    # Change shape from [batch_size] to [1, batch_size, 1]
    if batch_size == 1 and batch_first == False:
        tgt = tgt.unsqueeze(0)
        #tgt=tgt.unsqueeze(0) # change from [1] to [1, 1, 1]

    # Change shape from [batch_size] to [1, batch_size, 1]
    if batch_first == False and batch_size > 1:
        tgt = tgt.unsqueeze(0).unsqueeze(-1)

    # Create masks

    if tgt.shape[1] != 1:
      tgt=tgt.permute(1,0,2)

    dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]

    dim_b = src.shape[1] if batch_first == True else src.shape[0]

    #print('dimensiones',dim_a,dim_b)# 1, 35

    tgt_mask = generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_a,
        )

    src_mask = generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_b,
        )

    # Make final prediction

    src_mask,tgt_mask=src_mask.to(device),tgt_mask.to(device)

    #print()

    #print('tgt',tgt.shape)   # (8,1,4)
    #print('src',src.shape)   # (35,1,4)
    #print('src_mask',src_mask.shape)   # (8,35)
    #print('tgt_mask',tgt_mask.shape)   # (8,8)

    src=src.permute(1,0,2)
    tgt=tgt.permute(1,0,2)

    final_prediction = model(src, tgt, src_mask, tgt_mask)

    final_prediction = final_prediction.detach()

    if final_prediction.shape[1]>1:
      final_prediction=final_prediction[:,-1,:]
      final_prediction=final_prediction.unsqueeze(0)

    #print(final_prediction.shape) #1,1,4

    return final_prediction

def get_indices_input_target(num_obs, input_len, step_size, forecast_horizon, target_len):
        """
        Produce all the start and end index positions of all sub-sequences.
        The indices will be used to split the data into sub-sequences on which
        the models will be trained.
        Returns a tuple with four elements:
        1) The index position of the first element to be included in the input sequence
        2) The index position of the last element to be included in the input sequence
        3) The index position of the first element to be included in the target sequence
        4) The index position of the last element to be included in the target sequence

        Args:
            num_obs (int): Number of observations in the entire dataset for which
                            indices must be generated.
            input_len (int): Length of the input sequence (a sub-sequence of
                             of the entire data sequence)
            step_size (int): Size of each step as the data sequence is traversed.
                             If 1, the first sub-sequence will be indices 0-input_len,
                             and the next will be 1-input_len.
            forecast_horizon (int): How many index positions is the target away from
                                    the last index position of the input sequence?
                                    If forecast_horizon=1, and the input sequence
                                    is data[0:10], the target will be data[11:taget_len].
            target_len (int): Length of the target / output sequence.
        """

        input_len = round(input_len) # just a precaution
        start_position = 0
        stop_position = num_obs-1 # because of 0 indexing

        subseq_first_idx = start_position
        subseq_last_idx = start_position + input_len
        target_first_idx = subseq_last_idx + forecast_horizon
        target_last_idx = target_first_idx + target_len
        print("target_last_idx is {}".format(target_last_idx))
        print("stop_position is {}".format(stop_position))
        indices = []
        while target_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_last_idx, target_first_idx, target_last_idx))
            subseq_first_idx += step_size
            subseq_last_idx += step_size
            target_first_idx = subseq_last_idx + forecast_horizon
            target_last_idx = target_first_idx + target_len

        return indices

def get_indices_entire_sequence(data: pd.DataFrame, window_size: int, step_size: int) -> list:
        """
        Produce all the start and end index positions that is needed to produce
        the sub-sequences.
        Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
        sequence. These tuples should be used to slice the dataset into sub-
        sequences. These sub-sequences should then be passed into a function
        that slices them into input and target sequences.

        Args:
            num_obs (int): Number of observations (time steps) in the entire
                           dataset for which indices must be generated, e.g.
                           len(data)
            window_size (int): The desired length of each sub-sequence. Should be
                               (input_sequence_length + target_sequence_length)
                               E.g. if you want the model to consider the past 100
                               time steps in order to predict the future 50
                               time steps, window_size = 100+50 = 150
            step_size (int): Size of each step as the data sequence is traversed
                             by the moving window.
                             If 1, the first sub-sequence will be [0:window_size],
                             and the next will be [1:window_size].
        Return:
            indices: a list of tuples
        """

        stop_position = len(data)-1 # 1- because of 0 indexing

        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0

        subseq_last_idx = window_size

        indices = []

        while subseq_last_idx <= stop_position:

            indices.append((subseq_first_idx, subseq_last_idx))

            subseq_first_idx += step_size

            subseq_last_idx += step_size

        return indices

def train_model(model, training_dataloader, criterion, optimizer,scheduler, num_epochs,name,forecast_window,enc_seq_len,grad_norm_clip):
    best_loss = 100.0
    start=time.time()
    trigger_times=0
    patience=num_epochs//2
    device='cuda'
    #device='cpu' if torch.cuda.is_available else 'cuda'
    #print(device)

    print('Training model: ',name)

    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch nº: {epoch+1}/{num_epochs}")
        epoch_loss=0.0

        for i,(src,tgt_y) in tqdm(enumerate(training_dataloader),total=len(training_dataloader)):
          src=src[:,:,None]
          optimizer.zero_grad(set_to_none=True)

          tgt_mask=generate_square_subsequent_mask(
              dim1=forecast_window,
              dim2=forecast_window,
          )

          src_mask=generate_square_subsequent_mask(
             dim1=forecast_window,
             dim2=enc_seq_len
          )

          src,tgt_y,src_mask,tgt_mask=src.to(device),tgt_y.to(device),src_mask.to(device),tgt_mask.to(device)


          prediction=model(src,src_mask,tgt_mask)
          loss=criterion(prediction.to(device),tgt_y[:,None].float())

          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
          optimizer.step()


          epoch_loss += loss.item() * len(prediction)


        data_size = len(training_dataloader.dataset)
        epoch_loss = round(epoch_loss / data_size,4)
        print(f'Epoch {epoch + 1}/{num_epochs} | train | Loss: {epoch_loss:.4f}') #| Val: {epoch_acc:.4f}')



        if epoch_loss < best_loss:
          print('Loss decrease')
          torch.save(model.state_dict(), f'modelos/{name}')
          best_loss = epoch_loss

        else:
          trigger_times+=1
          if trigger_times>=patience:
            print('Model was early stopped')
            break

        if scheduler!=False:
          scheduler.step()


    print('Training finished')
    print('Training time: ',round(time.time()-start,2),'s')
    print('')
    return round(best_loss,4)

class TransformerDataset(torch.utils.data.Dataset):
    """
    Dataset class used for transformer models.

    """
    def __init__(self,
        data: torch.tensor,
        indices: list,
        enc_seq_len: int,
        dec_seq_len: int,
        target_seq_len: int
        ) -> None:

        """
        Args:
            data: tensor, the entire train, validation or test data sequence
                        before any slicing. If univariate, data.size() will be
                        [number of samples, number of variables]
                        where the number of variables will be equal to 1 + the number of
                        exogenous variables. Number of exogenous variables would be 0
                        if univariate.
            indices: a list of tuples. Each tuple has two elements:
                     1) the start index of a sub-sequence
                     2) the end index of a sub-sequence.
                     The sub-sequence is split into src, trg and trg_y later.
            enc_seq_len: int, the desired length of the input sequence given to the
                     the first layer of the transformer model.
            target_seq_len: int, the desired length of the target sequence (the output of the model)
            target_idx: The index position of the target variable in data. Data
                        is a 2D tensor
        """

        super().__init__()

        self.indices = indices

        self.data = data

        print("From get_src_trg: data size = {}".format(data.size()))

        self.enc_seq_len = enc_seq_len

        self.dec_seq_len = dec_seq_len

        self.target_seq_len = target_seq_len



    def __len__(self):

        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """
        # Get the first element of the i'th tuple in the list self.indicesasdfas
        start_idx = self.indices[index][0]

        # Get the second (and last) element of the i'th tuple in the list self.indices
        end_idx = self.indices[index][1]

        sequence = self.data[start_idx:end_idx]

        #print("From __getitem__: sequence length = {}".format(len(sequence)))

        src, trg, trg_y = self.get_src_trg(
            sequence=sequence,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len=self.target_seq_len
            )

        return src, trg, trg_y

    def get_src_trg(
        self,
        sequence: torch.Tensor,
        enc_seq_len: int,
        target_seq_len: int
        ):

        """
        Generate the src (encoder input), trg (decoder input) and trg_y (the target)
        sequences from a sequence.
        Args:
            sequence: tensor, a 1D tensor of length n where
                    n = encoder input length + target sequence length
            enc_seq_len: int, the desired length of the input to the transformer encoder
            target_seq_len: int, the desired length of the target sequence (the
                            one against which the model output is compared)
        Return:
            src: tensor, 1D, used as input to the transformer model
            trg: tensor, 1D, used as input to the transformer model
            trg_y: tensor, 1D, the target sequence against which the model output
                is compared when computing loss.
        """
        assert len(sequence) == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"

        # encoder input
        src = sequence[:enc_seq_len]

        # decoder input. As per the paper, it must have the same dimension as the
        # target sequence, and it must contain the last value of src, and all
        # values of trg_y except the last (i.e. it must be shifted right by 1)
        trg = sequence[enc_seq_len-1:len(sequence)-1]

        assert len(trg) == target_seq_len, "Length of trg does not match target sequence length"

        # The target sequence against which the model output will be compared to compute loss
        trg_y = sequence[-target_seq_len:]

        assert len(trg_y) == target_seq_len, "Length of trg_y does not match target sequence length"

        return src, trg, trg_y.squeeze(-1) # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len]
    
