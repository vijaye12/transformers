from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction, PatchTSMixerForPretraining
import os
import math
import tempfile
import torch
import torchinfo
# Third Party
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsfm_public.toolkit.time_series_preprocessor import get_datasets
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

def plot_preds(trainer, dset, plot_dir, num_plots=10, plot_prefix="valid", channel=-1):
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    random_indices = np.random.choice(len(dset), size=num_plots, replace=False)
    random_samples = torch.stack([dset[i]["past_values"] for i in random_indices])
    trainer.model = trainer.model.to(device)
    output = trainer.model(random_samples.to(device=device))
    
    y_hat = output.prediction_outputs[:, :, channel].detach().cpu().numpy()
    mask = output.mask[:, :, channel].detach().cpu().numpy()
    pred_len = y_hat.shape[1]

    # Set a more beautiful style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Adjust figure size and subplot spacing
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 20))
    for i, ri in enumerate(random_indices):
        batch = dset[ri]
        y = batch["past_values"][:pred_len, channel].squeeze().cpu().numpy()
        
        # Plot predicted values with a dashed line
        y_hat_plot = y_hat[i, ...] # np.concatenate((x, y_hat[i, ...]), axis=0)
        axs[i].plot(y_hat_plot, label="Reconstructed", linestyle="--", color="orange", linewidth=2)

        # Plot true values with a solid line
        axs[i].plot(y, label="Original", linestyle="-", color="blue", linewidth=2)

        axs[i].plot(mask[i, ...], label="mask", marker='o', linestyle='', color="red")

        # # Plot horizon border
        # axs[i].axvline(x=2 * pred_len, color="r", linestyle="-")

        axs[i].set_title(f"Example {random_indices[i]}")
        axs[i].legend()

    # Adjust overall layout
    plt.tight_layout()

    # Save the plot
    plot_filename = f"data_{plot_prefix}_ch_{str(channel)}.jpg"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, plot_filename))
    plt.show()



SEED = 42
set_seed(SEED)

# get data: 
target_dataset = "etth1"
DATA_ROOT_PATH = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"

# Results dir
OUT_DIR = "/dccstor/tsfm-irl/vijaye12/opensource/hack/"

dataset_path = DATA_ROOT_PATH
timestamp_column = "date"
id_columns = []
context_length = 96
target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
split_config = {
    "train": [0, 12 * 30 * 24],
    "valid": [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24],
    "test": [12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
}

data = pd.read_csv(
    dataset_path,
    parse_dates=[timestamp_column],
)

column_specifiers = {
    "timestamp_column": timestamp_column,
    "id_columns": id_columns,
    "target_columns": target_columns,
    "control_columns": [],
}


forecast_horizon = 1
patch_length = 8
batch_size = 64
num_workers = 4

tsp = TimeSeriesPreprocessor(
    **column_specifiers,
    context_length=context_length,
    prediction_length=forecast_horizon,
    scaling=True,
    encode_categorical=False,
    scaler_type="standard",
)

train_dataset, valid_dataset, test_dataset = get_datasets(tsp, 
                                                            data, 
                                                            split_config=split_config, 
                                                            )

# replace train_dataset, valid_dataset, test_dataset with the exact same datasets used in timesnet 
print(f"Data lengths: train = {len(train_dataset)}, val = {len(valid_dataset)}, test = {len(test_dataset)}")

config = PatchTSMixerConfig(
    context_length=context_length,
    prediction_length=forecast_horizon,
    patch_length=patch_length,
    num_input_channels=tsp.num_input_channels,
    patch_stride=patch_length,
    d_model=24,
    num_layers=8,
    expansion_factor=2,
    dropout=0.05,
    head_dropout=0.05,
    mode="mix_channel",  # change it `mix_channel` if we need to explicitly model channel correlations
    scaling="std",
    masked_loss = True,
    mask_type = "point",
    random_mask_ratio = 0.5, # change 0.125,0.25, .375, 0.5
)

model = PatchTSMixerForPretraining(config=config)

print(model)

print_col_names = [
        "num_params",
        "params_percent",
        "kernel_size",
        "trainable",
]

full_print_col_names = [
    "input_size",
    "output_size",
    "mult_adds",
] + print_col_names


model_config = model.config.to_dict()

print(torchinfo.summary(
                        model,
                        (1,context_length,tsp.num_input_channels),
                        depth=6,
                        col_names=full_print_col_names,
                        row_settings=["var_names"],
                        verbose=0,
                    ))



train_args = TrainingArguments(
    output_dir="./checkpoint/patchtsmixer/direct/train/output/",
    overwrite_output_dir=True,
    learning_rate=0.002, # 0.0001,
    num_train_epochs=100,
    do_eval=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    dataloader_num_workers=num_workers,
    report_to="tensorboard",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=3,
    logging_dir="./checkpoint/patchtsmixer/direct/train/logs/",  # Make sure to specify a logging directory
    load_best_model_at_end=True,  # Load the best model when training ends
    metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
    greater_is_better=False,  # For loss
    seed = SEED,
    
    # label_names=["future_values"],
)

# Create a new early stopping callback with faster convergence properties
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
    early_stopping_threshold=0.0001,  # Minimum improvement required to consider as improvement
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[early_stopping_callback],
)

print("\n\nDoing forecasting training on Etth1/train")
trainer.train()


result = trainer.evaluate(test_dataset)
print(result)



plot_preds(trainer=trainer, dset=test_dataset, plot_dir=os.path.join(OUT_DIR, "ettm2"), plot_prefix="test", channel=0)
