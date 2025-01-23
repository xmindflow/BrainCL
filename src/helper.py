import torch
from torch import autocast
import argparse
import os
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import DataLoader
from torch._dynamo import OptimizedModule
from glob import glob
from copy import deepcopy
from dataset.basic_dataset import BasicDataset


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=400, help="Number of epochs to train for"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--drop_modality", type=int, default=1, help="Randomly drop modalities"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count() - 1,
        help="Number of workers for dataloader",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to save/load checkpoint"
    )
    parser.add_argument(
        "--load_checkpoint_path_episode",
        type=str,
        default=None,
        help="Path to load checkpoint from (episode)",
    )
    parser.add_argument(
        "--load_checkpoint_path_epoch",
        type=str,
        default=None,
        help="Path to load checkpoint from (epoch)",
    )
    parser.add_argument(
        "--continue_training_from_episode",
        type=int,
        default=0,
        help="Continue training from episode checkpoint",
    )
    parser.add_argument(
        "--continue_training_from_epoch",
        type=int,
        default=0,
        help="Continue training from epoch checkpoint",
    )

    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/hdd/Continual_learning_data/FINAL",
        help="Root directory of data",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the experiment (used for saving checkpoints)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer to use"
    )
    parser.add_argument(
        "--show_progress", type=int, default=1, help="Show progress bar"
    )
    parser.add_argument("--beta", type=float, default=0.8, help="Beta for Cosine Loss")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for KL Loss")
    parser.add_argument("--amp", type=int, default=1, help="Use AMP")
    parser.add_argument("--compile", type=int, default=0, help="Compile model")
    parser.add_argument(
        "--num_experts", type=int, default=4, help="Number of experts in the model"
    )
    parser.add_argument(
        "--context_dim", type=int, default=10, help="Dimension of context vector"
    )
    parser.add_argument("--dynamic_coef", type=int, default=1, help="Use dynamic coef")
    parser.add_argument(
        "--tempurature", type=float, default=1.0, help="Tempurature for KL Loss"
    )
    parser.add_argument(
        "--alpha_max",
        type=float,
        default=1.0,
        help="Max value for alpha in dynamic coef",
    )
    parser.add_argument(
        "--sequence", type=int, default=0, help="which sequence to use for training"
    )
    return parser.parse_args()


def print_args(args):
    print("*" * 10)
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("*" * 10)


def get_optimizer(optimizer_name, model, lr):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.99, weight_decay=1e-5, nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def evaluate_model(test_datasets, model, device, dataloader_kwargs, amp, single_dataset=None):
    """
    Evaluates the model on either multiple test datasets or a single dataset.
    If `single_dataset` is provided, it overrides `test_datasets`.
    """
    
    dice_metric = DiceMetric(
        include_background=True, reduction="mean", get_not_nans=False
    )
    results = {}
    model.eval()
    model = model.to(device)
    
    if single_dataset is not None:
        test_datasets = [single_dataset]
    
    
    for test_dataset in test_datasets:
        dataset_name = test_dataset._get_name()
        print(f"Evaluating on {dataset_name}")

        tmp = test_dataset[0]
        has_context = len(tmp) == 3

        dice_metric.reset()
        dataloader = DataLoader(test_dataset, batch_size=1, **dataloader_kwargs)
        
        for data in tqdm(dataloader):
            img, gt = data[:2]
            img, gt = img.to(device), gt.to(device)
            
            inference_kwargs = {
                "inputs": img,
                "roi_size": (128, 128, 128),
                "sw_batch_size": 8,
                "predictor": model,
                "device": device,
                "overlap": 0.5,
                "sw_device": device,
            }
            if has_context:
                context = data[2].to(device)
                inference_kwargs["context"] = context # Only add this key if context exists
                
            with torch.inference_mode(), autocast(device.type, enabled=True) if amp else dummy_context():
                pred_logits: torch.Tensor = sliding_window_inference(**inference_kwargs)
                
                pred = pred_logits.sigmoid() > 0.5 if pred_logits.shape[1] == 1 else pred_logits.argmax(dim=1).unsqueeze(1)

                current_dice=dice_metric(pred, gt)
        
        del dataloader
        metric = dice_metric.aggregate().item()
        results[dataset_name] = metric
        print(f"Dice score on {dataset_name}: {metric}")
        dice_metric.reset()
        print("Evaluation completed")
    
    return results if single_dataset is None else metric


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def tmp_evaluation(train_dataset_orig, model, device, dataloader_kwargs, amp):
    ############################################################
    # this is dirty, fix later
    train_dataset: BasicDataset = deepcopy(train_dataset_orig)
    train_dataset.test_mode = True
    train_dataset.transform = train_dataset._get_transforms(test_mode=True)
    ###########################################################################
    metric = evaluate_model(
        test_datasets=None,  # No test datasets
        model=model,
        device=device,
        dataloader_kwargs=dataloader_kwargs,
        amp=amp,
        single_dataset=train_dataset,  # Pass the transformed dataset
    )
    print("TMP Evaluation completed!!!!!!!!!")
    return metric


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def save_checkpoint(
    model,
    teacher,
    optimizer,
    epoch_idx,
    episode_idx,
    grad_scaler,
    args,
    experiment_name,
    save_epoch: bool = False,
):
    if teacher is not None:
        teacher_to_be_saved = (
            teacher._orig_mod.state_dict()
            if isinstance(teacher, OptimizedModule)
            else teacher.state_dict()
        )
    else:
        teacher_to_be_saved = None

    checkpoint = {
        "model": (
            model._orig_mod.state_dict()
            if isinstance(model, OptimizedModule)
            else model.state_dict()
        ),
        "teacher": teacher_to_be_saved,
        "optimizer": optimizer.state_dict(),
        "episode_idx": episode_idx + 1,
        "epoch": epoch_idx + 1 if epoch_idx is not None else None,
        "grad_scaler": grad_scaler.state_dict() if grad_scaler is not None else None,
    }
    checkpoint_path = os.path.join(args.checkpoint_path, experiment_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    if save_epoch:
        files_to_delete = glob(
            os.path.join(checkpoint_path, f"exprience_{episode_idx}_epoch_*.pt")
        )
        for file in files_to_delete:
            os.remove(file)
        torch.save(
            checkpoint,
            os.path.join(
                checkpoint_path, f"exprience_{episode_idx}_epoch_{epoch_idx}.pt"
            ),
        )
    else:
        torch.save(
            checkpoint, os.path.join(checkpoint_path, f"exprience_{episode_idx}.pt")
        )
