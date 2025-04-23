import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ------------------------------
# Dataset Generation using HDF5
# ------------------------------
class FlowFieldDataset(Dataset):
    def __init__(self,
                 low_h5_path,
                 high_h5_path,
                 num_samples=None,
                 return_coords=False,
                 normalize=True,
                 mean_std_file='mean_std.npz',
                 x_range=(1.381, 1.881),
                 y_range=(0.0, 0.5)):
        """
        low_h5_path: Path to the HDF5 file for low-resolution flow field data.
        high_h5_path: Path to the HDF5 file for high-resolution flow field data.
        num_samples: Use the first num_samples samples; if None, use all samples.
        return_coords: Whether to return original X, Y coordinates (for visualization).
        normalize: Whether to normalize the (U, V) features.
        mean_std_file: File name to save/load mean and standard deviation values.
        x_range: Tuple (min_x, max_x) for filtering the X coordinate. Default is (1.381, 1.881).
        y_range: Tuple (min_y, max_y) for filtering the Y coordinate. Default is (0.0, 0.5).
        """
        # Save filtering ranges
        self.x_range = x_range
        self.y_range = y_range

        # Open HDF5 files
        self.low_h5 = h5py.File(low_h5_path, 'r')
        self.high_h5 = h5py.File(high_h5_path, 'r')

        # Get all sample keys (assuming each sample is stored as a dataset, e.g., 'sample_0', 'sample_1', ...)
        self.low_keys = sorted(list(self.low_h5.keys()))
        self.high_keys = sorted(list(self.high_h5.keys()))
        if num_samples is not None:
            self.low_keys = self.low_keys[:num_samples]
            self.high_keys = self.high_keys[:num_samples]
        assert len(self.low_keys) == len(self.high_keys), "Mismatch in number of low and high resolution samples!"

        # Read the first sample to filter rows based on coordinates and determine valid row counts
        low_sample = self.low_h5[self.low_keys[0]][:]
        high_sample = self.high_h5[self.high_keys[0]][:]

        # Filter low resolution sample based on X and Y range
        mask_low = (low_sample[:, 0] >= self.x_range[0]) & (low_sample[:, 0] <= self.x_range[1]) & \
                   (low_sample[:, 1] >= self.y_range[0]) & (low_sample[:, 1] <= self.y_range[1])
        low_sample_filtered = low_sample[mask_low]

        # Filter high resolution sample based on X and Y range
        mask_high = (high_sample[:, 0] >= self.x_range[0]) & (high_sample[:, 0] <= self.x_range[1]) & \
                    (high_sample[:, 1] >= self.y_range[0]) & (high_sample[:, 1] <= self.y_range[1])
        high_sample_filtered = high_sample[mask_high]

        # Record the number of rows after filtering
        self.N_low = low_sample_filtered.shape[0]
        self.N_high = high_sample_filtered.shape[0]

        # Record total number of columns in each sample
        self.total_input_dim = low_sample.shape[1]
        self.total_output_dim = high_sample.shape[1]

        # For training, assume features exclude the first 3 columns (e.g., X, Y, cellIndex)
        self.input_dim = self.total_input_dim - 3
        self.output_dim = self.total_output_dim - 3

        self.return_coords = return_coords
        self.normalize = normalize
        self.mean_std_file = mean_std_file

        # If normalization is enabled, compute or load the mean and standard deviation for normalization
        if self.normalize:
            if os.path.exists(self.mean_std_file):
                data = np.load(self.mean_std_file)
                self.mean_in = data['mean_in']  # shape: (2,)
                self.std_in = data['std_in']  # shape: (2,)
                self.mean_out = data['mean_out']
                self.std_out = data['std_out']
            else:
                self.mean_in, self.std_in, self.mean_out, self.std_out = self._compute_mean_std()
                np.savez(self.mean_std_file,
                         mean_in=self.mean_in,
                         std_in=self.std_in,
                         mean_out=self.mean_out,
                         std_out=self.std_out)
        else:
            self.mean_in = np.zeros(2, dtype=np.float32)
            self.std_in = np.ones(2, dtype=np.float32)
            self.mean_out = np.zeros(2, dtype=np.float32)
            self.std_out = np.ones(2, dtype=np.float32)

    def _compute_mean_std(self):
        """
        Traverse all samples to compute the mean and standard deviation of the (U, V) features.
        Here we assume that normalization is applied on columns 3 and 4 (indices 2:4).
        """
        all_in = []
        all_out = []
        for lk, hk in zip(self.low_keys, self.high_keys):
            low_data = self.low_h5[lk][:]
            high_data = self.high_h5[hk][:]
            mask_low = (low_data[:, 0] >= self.x_range[0]) & (low_data[:, 0] <= self.x_range[1]) & \
                       (low_data[:, 1] >= self.y_range[0]) & (low_data[:, 1] <= self.y_range[1])
            low_data_filtered = low_data[mask_low]
            mask_high = (high_data[:, 0] >= self.x_range[0]) & (high_data[:, 0] <= self.x_range[1]) & \
                        (high_data[:, 1] >= self.y_range[0]) & (high_data[:, 1] <= self.y_range[1])
            high_data_filtered = high_data[mask_high]
            # Extract U, V columns (assumed to be the 3rd and 4th columns)
            all_in.append(low_data_filtered[:, 2:4])
            all_out.append(high_data_filtered[:, 2:4])
        all_in = np.concatenate(all_in, axis=0)
        all_out = np.concatenate(all_out, axis=0)
        mean_in = np.mean(all_in, axis=0)
        std_in = np.std(all_in, axis=0) + 1e-8  # Prevent division by zero
        mean_out = np.mean(all_out, axis=0)
        std_out = np.std(all_out, axis=0) + 1e-8
        return mean_in, std_in, mean_out, std_out

    def __len__(self):
        return len(self.low_keys)

    def __getitem__(self, idx):
        # Load low-resolution and high-resolution data from HDF5 files
        low_data = self.low_h5[self.low_keys[idx]][:]
        high_data = self.high_h5[self.high_keys[idx]][:]
        # Filter rows based on x_range and y_range
        mask_low = (low_data[:, 0] >= self.x_range[0]) & (low_data[:, 0] <= self.x_range[1]) & \
                   (low_data[:, 1] >= self.y_range[0]) & (low_data[:, 1] <= self.y_range[1])
        low_data_filtered = low_data[mask_low]
        mask_high = (high_data[:, 0] >= self.x_range[0]) & (high_data[:, 0] <= self.x_range[1]) & \
                    (high_data[:, 1] >= self.y_range[0]) & (high_data[:, 1] <= self.y_range[1])
        high_data_filtered = high_data[mask_high]

        # Convert data to torch tensors
        low_tensor = torch.tensor(low_data_filtered, dtype=torch.float32)
        high_tensor = torch.tensor(high_data_filtered, dtype=torch.float32)

        # Extract (U, V) features (assumed to be columns 3 and 4)
        low_features = low_tensor[:, 2:4]
        high_features = high_tensor[:, 2:4]

        # (Optional) Normalize the features
        if self.normalize:
            for c in range(2):
                low_features[:, c] = (low_features[:, c] - self.mean_in[c]) / self.std_in[c]
            for c in range(2):
                high_features[:, c] = (high_features[:, c] - self.mean_out[c]) / self.std_out[c]

        # (Optional) Return original coordinates for visualization
        if self.return_coords:
            coords_low = low_tensor[:, :2]
            coords_high = high_tensor[:, :2]
            return low_features, high_features, coords_low, coords_high
        else:
            return low_features, high_features

    def __del__(self):
        # Close HDF5 files to free up resources
        self.low_h5.close()
        self.high_h5.close()
