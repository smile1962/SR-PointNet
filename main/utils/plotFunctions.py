import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from matplotlib import cm
import cv2

def plot_contour_save(X, Y, values,
                    title, xlabel, ylabel, colorbar_label,
                    vmin, vmax, save_path,
                    grid_x=1000, grid_y=400,
                    cmap='jet'):
    """
    Plot a cloud (image-style) map via imshow, align the colorbar height with the main axes, and save the figure.

    Parameters:
        X (array-like): 1D array of X coordinates.
        Y (array-like): 1D array of Y coordinates.
        values (array-like): Data values at each (X, Y) point.
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        colorbar_label (str): Label for the colorbar.
        vmin (float): Minimum value for the colormap.
        vmax (float): Maximum value for the colormap.
        save_path (str): Path where the figure will be saved.
        grid_x (int, optional): Number of grid points in X direction. Defaults to 1000.
        grid_y (int, optional): Number of grid points in Y direction. Defaults to 400.
        cmap (str, optional): Colormap name. Defaults to 'jet'.
    """
    # 1) generate uniform grid in X/Y
    xi = np.linspace(np.min(X), np.max(X), grid_x)
    yi = np.linspace(0, 0.5, grid_y)
    Xi, Yi = np.meshgrid(xi, yi)

    # 2) interpolate scattered data onto grid
    Zi = griddata((X, Y), values, (Xi, Yi), method='linear')

    # 3) prepare imshow options
    plot_options = {
        'cmap':   cmap,
        'origin': 'lower',
        'extent': [X.min(), X.max(), Y.min(), Y.max()],
        'vmin':   vmin,
        'vmax':   vmax,
        'aspect': 'equal'
    }

    # 4) create figure and main axes
    fig, ax = plt.subplots()

    # 5) plot image
    im = ax.imshow(Zi, **plot_options)

    # 6) set titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', which='major', direction='in', length=1, width=0.5)

    # 7) add colorbar with same height as main axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(colorbar_label)
    cbar.ax.tick_params(axis='y', direction='in', length=1, width=0.5)

    # 8) save and close
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def plot_scatter_save(X, Y, values, title, xlabel, ylabel, colorbar_label,
                      vmin, vmax, scatter_size, save_path, aspect='equal'):
    """
    Plot a scatter plot and save the figure to a file.

    Parameters:
        X (array-like): 1D array of X coordinates.
        Y (array-like): 1D array of Y coordinates.
        values (array-like): Data values used for coloring the scatter plot.
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        colorbar_label (str): Label for the colorbar.
        vmin (float): Minimum value for the colormap.
        vmax (float): Maximum value for the colormap.
        scatter_size (float): Size of the scatter points.
        save_path (str): Path where the figure will be saved.
        aspect (str, optional): Aspect ratio of the plot. Defaults to 'equal'.
        cbar_height (float, optional): Fixed height of the colorbar. Defaults to 0.8.
    """
    # Create a new figure and get the current axes
    fig, ax = plt.subplots()

    sc = ax.scatter(X, Y, c=values, cmap='jet', s=scatter_size, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', which='major', direction='in', length=1, width=0.5)

    ax.set_aspect(aspect, adjustable='box')
    # ax.set_xlim(np.min(X), np.max(X))
    # ax.set_ylim(np.min(Y), np.max(Y))
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(0, 0.5)

    # === Colorbar with same height as main axis ===
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label(colorbar_label)
    cbar.ax.tick_params(axis='y', direction='in', length=1, width=0.5)

    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def calculate_rmse_mae(y_true, y_pred):
    """
    Calculate RMSE and MAE between true and predicted values.

    Parameters:
        y_true (array-like): Ground truth values, shape (n_samples,)
        y_pred (array-like): Predicted values, shape (n_samples,)

    Returns:
        rmse (float): Root Mean Square Error
        mae (float): Mean Absolute Error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Calculate MAE
    mae = np.mean(np.abs(y_true - y_pred))

    return rmse, mae

def filter_by_x(pred_full, gt_full, a, tol=1e-5):
    """
    Filter the rows of the predicted and ground truth data where the X coordinate equals a.

    Both pred_full and gt_full are expected to be numpy arrays with shape [N, 4], where:
        - Column 0: X coordinates
        - Column 1: Y coordinates
        - Column 2: U component (velocity in X direction)
        - Column 3: V component (velocity in Y direction)

    Parameters:
        pred_full (np.ndarray): Array containing predicted data.
        gt_full (np.ndarray): Array containing ground truth data.
        a (float): The X coordinate value to filter by.
        tol (float, optional): Tolerance for floating-point comparisons. Defaults to 1e-5.

    Returns:
        filtered_pred (np.ndarray): Filtered predicted data with X == a.
        filtered_gt (np.ndarray): Filtered ground truth data with X == a.
    """
    # Use np.isclose to account for floating point inaccuracies when comparing X values
    mask_pred = np.isclose(pred_full[:, 0], a, atol=tol)
    mask_gt = np.isclose(gt_full[:, 0], a, atol=tol)

    filtered_pred = pred_full[mask_pred]
    filtered_gt = gt_full[mask_gt]

    return filtered_pred, filtered_gt


def sort_data_for_plot(x_values_pred, speed_pred):
    """
    对 x_values_pred 和 speed_pred 进行排序，保证 x_values_pred 升序排列，
    并且相应地重新排列 speed_pred。

    Parameters:
        x_values_pred (ndarray): 形状为 (401,) 的 x 轴数据
        speed_pred (ndarray): 形状为 (401,) 的 y 轴数据，对应 x_values_pred

    Returns:
        sorted_x (ndarray): 排序后的 x 轴数据（升序）
        sorted_speed (ndarray): 根据 x 轴排序后对应的 y 轴数据
    """
    # 获取 x_values_pred 升序排列的索引
    sorted_indices = np.argsort(x_values_pred)
    # 根据索引对 x_values_pred 和 speed_pred 进行排序
    sorted_x = x_values_pred[sorted_indices]
    sorted_speed = speed_pred[sorted_indices]

    return sorted_x, sorted_speed

def plot_dual_y_curve(filtered_pred, filtered_gt, title, xlabel, ylabel,
                      pred_label, gt_label, save_path):
    """
    Plot a dual-axis curve to compare the resultant speeds of the predicted and ground truth data
    along a specific cross-section.

    The input arrays (filtered_pred and filtered_gt) should have shape [M, 4]. Both arrays contain:
        - Column 0: X coordinate (identical for the filtered data)
        - Column 1: Y coordinate (used as the X-axis for the curve)
        - Column 2: U component (velocity in X direction)
        - Column 3: V component (velocity in Y direction)

    The resultant speed is computed as the square root of (U^2 + V^2).

    Parameters:
        filtered_pred (np.ndarray): Filtered predicted data.
        filtered_gt (np.ndarray): Filtered ground truth data.
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        pred_label (str): Label for the predicted speed curve (left Y-axis).
        gt_label (str): Label for the ground truth speed curve (right Y-axis).
        save_path (str): File path to save the plot.
    """
    # Compute resultant speeds for predicted and ground truth data
    speed_pred = np.sqrt(filtered_pred[:, 2] ** 2 + filtered_pred[:, 3] ** 2)
    speed_gt = np.sqrt(filtered_gt[:, 2] ** 2 + filtered_gt[:, 3] ** 2)

    # Use the second column (Y coordinates) as the X-axis for the curves
    x_values_pred = filtered_pred[:, 1]
    x_values_gt = filtered_gt[:, 1]

    # sort the data
    x_values_pred_sort, speed_pred_sort = sort_data_for_plot(x_values_pred, speed_pred)
    x_values_gt_sort, speed_gt_sort = sort_data_for_plot(x_values_gt, speed_gt)

    # calculate RMSE and MAE
    rmes, mae = calculate_rmse_mae(speed_gt_sort, speed_pred_sort)
    print(f"The RMSE and MAE values are {rmes} m/s and {mae} m/s.")

    # Create a new figure and primary axis
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel, color='black')  # Y轴标签字体使用黑色
    l1, = ax1.plot(x_values_pred_sort, speed_pred_sort, '-', color='tab:blue', label=pred_label)
    ax1.tick_params(axis='y', labelcolor='black', direction='in')  # Y轴刻度线向内

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel(gt_label, color='black')  # Y轴标签字体使用黑色
    l2, = ax2.plot(x_values_gt_sort, speed_gt_sort, '-', color='tab:red', label=gt_label)
    ax2.tick_params(axis='y', labelcolor='black', direction='in')  # Y轴刻度线向内

    # Set plot title and layout
    plt.title(title)
    fig.tight_layout()

    # Set the same y-axis limits for both axes
    min_speed = min(np.min(speed_pred), np.min(speed_gt))
    max_speed = max(np.max(speed_pred), np.max(speed_gt))
    ax1.set_ylim(min_speed, max_speed)
    ax2.set_ylim(min_speed, max_speed)

    # Hide the right y-axis labels
    ax2.yaxis.set_visible(False)

    # Add legend
    lines = [l1, l2]
    labels = [pred_label, gt_label]
    ax1.legend(lines, labels, loc='upper right')

    # Ensure the output directory exists and save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def plot_relative_error_histogram(Up, Ug, save_path, x_range=(0, 1)):
    """
    Plots a histogram of relative errors and saves it to the specified path.

    Parameters:
    Up -- Model predicted values, shape (n_samples,)
    Ug -- Ground truth values, shape (n_samples,)
    save_path -- Path to save the histogram, default is "../relative_error"
    """
    # Ensure inputs are NumPy arrays
    Up = np.asarray(Up)
    Ug = np.asarray(Ug)

    # Data preprocessing
    valid_indices = Ug >= 0.05
    Ug_filtered = Ug[valid_indices]
    Up_filtered = Up[valid_indices]

    # Calculate relative error
    relative_error = np.abs(Up_filtered - Ug_filtered) / (np.abs(Ug_filtered))

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(relative_error, bins=200, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Prediction of resultant velocity')
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set X-axis range
    # plt.xlim(x_range)

    # Save the plot
    plt.savefig(save_path)
    plt.close()

def plot_loss_curves(file_path, save_path="../../resultSaving"):
    """
    Plots training and validation loss curves from a log file and saves the plot.

    Parameters:
    file_path -- Path to the loss log file
    save_path -- Path to save the plot, default is "../../resultSaving"
    """
    # Read the log file
    data = pd.read_csv(file_path, header=0)

    # Extract data
    epochs = data.iloc[:, 0].values
    train_loss = data.iloc[:, 1].values
    val_loss = data.iloc[:, 2].values

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Set plot parameters
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    plt.figure(figsize=(10, 6))

    # Plot training and validation loss
    plt.plot(epochs, train_loss, label='Training Loss', color='blue', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linewidth=2)

    # Set axis ticks inward
    plt.tick_params(axis='both', direction='in')

    # Add legend, title, and labels
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curves saved to: {save_path}")

def export_to_excel(pred_full, gt_full, ori_full, save_path):
    """
    Export three ndarray variables to an Excel file with each variable in a separate sheet.

    Parameters:
        pred_full (ndarray): Predicted data, shape (N, 4)
        gt_full (ndarray): Ground truth data, shape (M, 4)
        ori_full (ndarray): Original data, shape (K, 4)
        save_path (str): Path to save the Excel file
    """
    # 确保输入是 NumPy 数组
    pred_full = np.asarray(pred_full)
    gt_full = np.asarray(gt_full)
    ori_full = np.asarray(ori_full)

    # 创建 DataFrame
    df_pred = pd.DataFrame(pred_full, columns=['X', 'Y', 'U', 'V'])
    df_gt = pd.DataFrame(gt_full, columns=['X', 'Y', 'U', 'V'])
    df_ori = pd.DataFrame(ori_full, columns=['X', 'Y', 'U', 'V'])

    # 使用 ExcelWriter 将数据写入 Excel 文件
    with pd.ExcelWriter(save_path) as writer:
        df_pred.to_excel(writer, sheet_name='Predicted', index=False)
        df_gt.to_excel(writer, sheet_name='Ground Truth', index=False)
        df_ori.to_excel(writer, sheet_name='Original', index=False)

    print(f"Data successfully exported to {save_path}")

def density_scatter(x, y, ax=None, bins=100, point_size=16):
    """Draw a density-colored scatter plot (helper function)"""
    if ax is None:
        fig, ax = plt.subplots()

    # Compute density
    data, x_e, y_e = np.histogram2d(x, y, bins=bins)
    z = np.log1p(data.T.ravel())

    # Map color index
    x_idx = np.clip(np.digitize(x, x_e) - 1, 0, bins-1)
    y_idx = np.clip(np.digitize(y, y_e) - 1, 0, bins-1)
    colors = z[x_idx * bins + y_idx]

    # Plot scatter with normalized colormap
    norm = plt.Normalize(vmin=np.percentile(colors, 5),
                         vmax=np.percentile(colors, 95))
    sc = ax.scatter(x, y, c=colors, s=point_size, alpha=0.6,
                   cmap=cm.plasma, norm=norm, edgecolors='none')

    # Configure colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Log density', fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    return ax

def plot_correlation_scatter(pred, truth, save_path, figsize=(10, 5), dpi=600):
    """
    Plot and save a correlation scatter plot

    Parameters:
        pred: Array of predicted values (ndarray)
        truth: Array of ground truth values (ndarray)
        save_path: Path to save the plot (str)
        figsize: Figure size (tuple, default (10,8))
        dpi: Resolution (int, default 600)
    """
    # Initialize figure settings
    plt.rcParams.update({
        'font.size': 24,
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'stix'
    })
    fig, ax = plt.subplots(figsize=figsize)

    # Draw density scatter
    density_scatter(pred, truth, ax=ax, bins=100, point_size=16)

    # Compute correlation and RMSE
    rho = np.corrcoef(pred, truth)[0, 1]
    rmse = np.sqrt(np.mean((pred - truth) ** 2))

    # Add reference line and annotation
    lim_min = min(pred.min(), truth.min())
    lim_max = max(pred.max(), truth.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=3)
    ax.text(0.05, 0.8, f'R = {rho:.3f}\nRMSE = {rmse:.3f} m/s',
            transform=ax.transAxes,
            fontdict={'family': 'Times New Roman', 'size': 24},
            bbox=dict(facecolor='white', alpha=0.8))

    # Set axis labels and appearance
    ax.set_xlabel('Predicted velocity (m/s)', fontsize=24)
    # ax.set_xlabel('Bicubic interpolation velocity (m/s)', fontsize=24)
    ax.set_ylabel('Numerical solution (m/s)', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.set_aspect('equal')
    ax.grid(alpha=0.2)

    # Create directories and save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Correlation plot saved to: {save_path}")

# def interpolate_lowres_to_highres_flat(
#     Xo_mask,
#     Yo_mask,
#     speed_ori_in_region,
#     grid_Xo,
#     grid_Yo,
#     grid_Xp,
#     grid_Yp
# ):
#     """
#     Interpolate low-resolution scattered data onto a regular grid,
#     upscale to high-resolution using bicubic interpolation,
#     and return the result as a flattened 1D array.
#
#     Parameters:
#         Xo_mask: 1D array of X coordinates of low-res data points
#         Yo_mask: 1D array of Y coordinates of low-res data points
#         speed_ori_in_region: 1D array of speed values at low-res points
#         grid_Xo: 1D array defining X axis of low-resolution grid
#         grid_Yo: 1D array defining Y axis of low-resolution grid
#         grid_Xp: 1D array defining X axis of high-resolution grid
#         grid_Yp: 1D array defining Y axis of high-resolution grid
#
#     Returns:
#         1D array of interpolated high-resolution values, flattened
#     """
#     # Create meshgrid for low-resolution grid
#     grid_xo_mesh, grid_yo_mesh = np.meshgrid(grid_Xo, grid_Yo)
#
#     # Interpolate scattered low-res data to low-res grid
#     lowres_grid = griddata(
#         points=(Xo_mask, Yo_mask),
#         values=speed_ori_in_region,
#         xi=(grid_xo_mesh, grid_yo_mesh),
#         method='linear',
#         fill_value=0.0
#     )
#
#     # Resize to high-resolution shape using bicubic interpolation
#     upsampled = cv2.resize(
#         lowres_grid,
#         dsize=(grid_Xp.size, grid_Yp.size),
#         interpolation=cv2.INTER_CUBIC
#     )
#
#     # Flatten and return
#     return upsampled.flatten()

def interpolate_lowres_to_highres_flat(
        Xo_mask,
        Yo_mask,
        speed_ori_in_region,
        grid_Xo,
        grid_Yo,
        grid_Xp,
        grid_Yp,
        visualize=False,
        save_path=None,
        dpi=300
):
    """
    Interpolate low-resolution scattered data onto a regular grid,
    upscale to high-resolution using bicubic interpolation,
    and return the result as a flattened 1D array.

    Parameters:
        Xo_mask: 1D array of X coordinates of low-res data points
        Yo_mask: 1D array of Y coordinates of low-res data points
        speed_ori_in_region: 1D array of speed values at low-res points
        grid_Xo: 1D array defining X axis of low-resolution grid
        grid_Yo: 1D array defining Y axis of low-resolution grid
        grid_Xp: 1D array defining X axis of high-resolution grid
        grid_Yp: 1D array defining Y axis of high-resolution grid
        visualize: bool, whether to generate validation plots
        save_path: str, directory to save visualization results
        dpi: int, output image resolution

    Returns:
        1D array of interpolated high-resolution values, flattened
    """
    # Create meshgrid for low-resolution grid
    grid_xo_mesh, grid_yo_mesh = np.meshgrid(grid_Xo, grid_Yo)

    # Interpolate scattered low-res data to low-res grid
    lowres_grid = griddata(
        points=(Xo_mask, Yo_mask),
        values=speed_ori_in_region,
        xi=(grid_xo_mesh, grid_yo_mesh),
        method='linear',
        fill_value=0.0
    )

    # Visualization: Low-res interpolation result
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(grid_xo_mesh, grid_yo_mesh, lowres_grid, shading='auto', cmap='jet')
        plt.colorbar(label='Velocity (m/s)')
        plt.title('Low-resolution Interpolation Result')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        if save_path:
            plt.savefig(f"{save_path}_lowres.png", dpi=dpi, bbox_inches='tight')
        plt.close()

    # Resize to high-resolution shape using bicubic interpolation
    upsampled = cv2.resize(
        lowres_grid,
        dsize=(grid_Xp.size, grid_Yp.size),
        interpolation=cv2.INTER_CUBIC
    )

    # Create high-res meshgrid for visualization
    grid_xp_mesh, grid_yp_mesh = np.meshgrid(grid_Xp, grid_Yp)

    # Visualization: High-res upsampled result
    if visualize:
        plt.figure(figsize=(12, 8))
        plt.pcolormesh(grid_xp_mesh, grid_yp_mesh, upsampled, shading='auto', cmap='jet')
        plt.colorbar(label='Velocity (m/s)')
        plt.title('High-resolution Upsampled Result')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        if save_path:
            plt.savefig(f"{save_path}_highres.png", dpi=dpi, bbox_inches='tight')
        plt.close()

    return upsampled.flatten()