import argparse

# def get_config():
#     parser = argparse.ArgumentParser(description="Flow Field Prediction Training")
#
#     # Basic training parameters
#     parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
#     parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
#     parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
#     parser.add_argument("--global_feat_dim", type=int, default=512, help="Global feature dimension for the encoder")
#     parser.add_argument("--use_samples", type=int, default=100, help="Number of samples to be used")
#     parser.add_argument("--train_ratio", type=float, default=0.6, help="Proportion of training set")
#     parser.add_argument("--test_ratio", type=float, default=0.2, help="Proportion of test set")
#     parser.add_argument("--val_ratio", type=float, default=0.2, help="Proportion of validation set")
#
#     # Data directories and normalization configuration
#     parser.add_argument("--low_res_folder", type=str, default=r"../dataset/case1LR_generalization_Geo.h5", help="Folder for low-resolution data")
#     parser.add_argument("--val_low_res_folder", type=str, default=r"../dataset/case1LR_generalization_Geo.h5", help="Folder for validation low-resolution data")
#     parser.add_argument("--high_res_folder", type=str, default=r"../dataset/case1HR_generalization_Geo.h5", help="Folder for high-resolution data")
#     parser.add_argument("--mean_std_file", type=str, default="mean_std_case1_val_Geo.npz", help="File name to save/load mean and std values")
#
#     # Model and weight saving paths
#     parser.add_argument("--save_dir", type=str, default="../PointnetWeights", help="Directory for saving model weights")
#     parser.add_argument("--best_save_dir", type=str, default="../PointnetWeights/best_case1_val_Geo.pth", help="Directory for saving the best model weights")
#     parser.add_argument("--final_save_dir", type=str, default="../PointnetWeights/case1_val_Geo.pth", help="Directory for saving the final model weights")
#     parser.add_argument("--save_root", type=str, default="../resultSaving_val_Geo", help="Directory for saving the validation results")
#
#     return parser.parse_args()

def get_config():
    parser = argparse.ArgumentParser(description="Flow Field Prediction Training")

    # Basic training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--global_feat_dim", type=int, default=512, help="Global feature dimension for the encoder")
    parser.add_argument("--use_samples", type=int, default=500, help="Number of samples to be used")
    parser.add_argument("--train_ratio", type=float, default=0.6, help="Proportion of training set")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Proportion of test set")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Proportion of validation set")

    # Data directories and normalization configuration
    parser.add_argument("--low_res_folder", type=str, default=r"../dataset/case2LR.h5", help="Folder for low-resolution data")
    parser.add_argument("--val_low_res_folder", type=str, default=r"../dataset/case2LR.h5", help="Folder for validation low-resolution data")
    parser.add_argument("--high_res_folder", type=str, default=r"../dataset/case2HR.h5", help="Folder for high-resolution data")
    parser.add_argument("--mean_std_file", type=str, default="mean_std_case2.npz", help="File name to save/load mean and std values")

    # Model and weight saving paths
    parser.add_argument("--save_dir", type=str, default="../PointnetWeights", help="Directory for saving model weights")
    parser.add_argument("--best_save_dir", type=str, default="../PointnetWeights/best_case2.pth", help="Directory for saving the best model weights")
    parser.add_argument("--final_save_dir", type=str, default="../PointnetWeights/case2.pth", help="Directory for saving the final model weights")
    parser.add_argument("--save_root", type=str, default="../resultSaving_case2", help="Directory for saving the validation results")

    return parser.parse_args()

if __name__ == "__main__":
    config = get_config()
    print(config)
