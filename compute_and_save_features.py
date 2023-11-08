import torch
import numpy as np
import pickle
from config import get_config
from agent import Agent
import torch
from tqdm import tqdm
import numpy as np
import sys

from dataset.dataset_modelnet import NonShufflingModelNetDataModule

def main():
    device = 'cuda'
    config = get_config("train")
    # Initialize your data module and agent (model)
    data_module = NonShufflingModelNetDataModule(config)
    agent = Agent(config)

    # Move the agent to the appropriate device and set to evaluation mode
    agent = agent.to(device)
    agent.eval()

    # Define paths to save features and metadata
    features_path_template = '{}_features.npz'  # {} is a placeholder for train/val/test
    metadata_path_template = '{}_metadata.pkl'

    # Process each dataset type
    for dataset_type in ['train', 'val', 'test']:
        # Setup the data module for the current stage
        if dataset_type in ['train']:
            data_module.setup(stage='fit')
        else:
            data_module.setup(stage='test')
        
        # Select the appropriate dataset
        if dataset_type == 'train':
            dataloader = data_module.train_dataloader()
        elif dataset_type == 'val':
            dataloader = data_module.val_dataloader()
        elif dataset_type == 'test':
            dataloader = data_module.test_dataloader()
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        # Compute features
        features = list([])
        metadata = list([])
        # Calculate the number of batches that make up approximately 5% of the dataset
        five_percent_batches = max(1, np.ceil(len(dataloader) * 0.05))

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc=f"Processing {dataset_type} data")):
                batch = {k: v.to(agent.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                # Compute features using the agent's 'compute_feature' method
                feature, _ = agent.compute_feature(batch, condition=config.condition)
                features.append(feature.cpu().numpy())

                # Save metadata (e.g., labels, img_ids)
                meta = {
                    'idx': batch['idx'].cpu().numpy(),
                    'cate': batch['cate'].cpu().numpy(),
                    'quat': batch['quat'].cpu().numpy(),
                    'rot_mat': batch['rot_mat'].cpu().numpy(),
                    'img_id': batch['img_id']
                }
                metadata.append(meta)
                # Check if we have processed another 5% of the data
                if (i + 1) % five_percent_batches == 0:
                    print(f"Saving intermediate results at {100 * (i + 1) / len(dataloader):.2f}% completion")
                    # Concatenate all features and metadata processed so far
                    partial_features = np.concatenate(features, axis=0)
                    partial_metadata = {key: np.concatenate([d[key] for d in metadata], axis=0) for key in metadata[0]}
                    print(f"Partial features shape: {partial_features.shape}")
                    print(f"Partial features storage: {sys.getsizeof(partial_features) / 1024 ** 3:.2f} GB")
                    # Save partial features and metadata to disk
                    partial_features_path = f'partial_{features_path_template.format(dataset_type)}'
                    partial_metadata_path = f'partial_{metadata_path_template.format(dataset_type)}'
                    np.savez_compressed(partial_features_path, features=partial_features)
                    with open(partial_metadata_path, 'wb') as f:
                        pickle.dump(partial_metadata, f)
                    del partial_features, partial_metadata

        # After the loop is finished, save the final complete set
        all_features = np.concatenate(features, axis=0)
        all_metadata = {key: np.concatenate([d[key] for d in metadata], axis=0) for key in metadata[0]}

        # Save final features and metadata to disk
        features_path = features_path_template.format(dataset_type)
        metadata_path = metadata_path_template.format(dataset_type)
        np.savez_compressed(features_path, features=all_features)
        with open(metadata_path, 'wb') as f:
            pickle.dump(all_metadata, f)


if __name__ == "__main__":
    main()