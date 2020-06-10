import errno
import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data

def create_embeddings(model_fc, ds, model_name, storage_path, storage_size=1000, parallel=True):
    """
    Takes in a feature extraction model and a dataset and stores embeddings in npz.
    model_fc: feature extraction model
    ds: dataloader
    model_name: name of the model
    storage_path: where to store embeddings
    storage_size: vector size of each .npz file
    parallel: enables data parallel modus
    """    

    # create folder when doesn't exist yet
    try:
        os.makedirs(storage_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    feature_extractor = model_fc
    if parallel:
        feature_extractor = nn.DataParallel(model_fc)
    target_dataset = ds
    len_target_dataset = len(target_dataset)
    # save some memory

    feature_extractor.eval()
    
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Moving model to {device}")
        feature_extractor = feature_extractor.to(device)
        params = {'batch_size': 50,
                  'shuffle': False,
                  'num_workers': 6,
                  'pin_memory': False}

        print(f"Length of dataset is {len_target_dataset}")
        if (len_target_dataset >= storage_size):

            if len_target_dataset % storage_size != 0:
                until_i = (len_target_dataset // storage_size + 1)
            else:
                until_i = (len_target_dataset // storage_size)

            for i in range(until_i):

                """Check if we overshot the entries"""
                if ((i+1)*storage_size <= len_target_dataset):
                    t_dataset = torch.utils.data.Subset(target_dataset, range(i*storage_size, (i+1)*storage_size))
                else:
                    remainder = len_target_dataset - i*storage_size
                    print(f"Calculating for remainder: {remainder} because we want to extract {(i+1)*storage_size}")
                    t_dataset = torch.utils.data.Subset(target_dataset, range(i*storage_size, (i*storage_size) + remainder))# use remainder

                training_generator = data.DataLoader(t_dataset, **params)

                features = torch.Tensor([]).to(device)
                labels = torch.LongTensor([]).to(device)

                for local_batch, local_labels in training_generator:
                    local_batch = local_batch.to(device)
                    local_labels = local_labels.to(device)
                    output = feature_extractor(local_batch)
                    features = torch.cat([features, output], dim=0)
                    labels = torch.cat([labels, local_labels], dim=0)

                print(features.size())
                features = features.to("cpu")
                labels = labels.to("cpu")

                x = features.detach().numpy()
                y = labels.detach().numpy()

                np.savez_compressed(f'{storage_path}/{model_name}_{i}.npz', x=x, y=y)

                del features
                del labels
                del local_batch
                del local_labels
                torch.cuda.empty_cache()

        if (len_target_dataset < storage_size):
            training_generator = data.DataLoader(target_dataset, **params)
            features = torch.Tensor([]).to(device)
            labels = torch.LongTensor([]).to(device)

            for local_batch, local_labels in training_generator:
                local_batch = local_batch.to(device)
                local_labels = local_labels.to(device)
                output = feature_extractor(local_batch)
                features = torch.cat([features, output], dim=0)
                labels = torch.cat([labels, local_labels], dim=0)

            print(features.size())
            features = features.to("cpu")
            labels = labels.to("cpu")

            x = features.detach().numpy()
            y = labels.detach().numpy()

            np.savez_compressed(f'{storage_path}/{model_name}_0.npz', x=x, y=y)

            del features
            del labels
            del local_batch
            del local_labels
            torch.cuda.empty_cache()