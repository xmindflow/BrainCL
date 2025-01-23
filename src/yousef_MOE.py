import torch
from torch import nn, autocast
import helper as hp
import sys
import os
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nets.MoE import MOEResUNetWithBottleneck
from torchinfo import summary
from monai.losses import DiceCELoss
from dataset.basic_dataset import BasicDataset
from monai.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from tqdm.auto import tqdm
from copy import deepcopy
import json
import torch.nn.functional as F
from monai.data import MetaTensor
from monai.utils import set_determinism
import time
from helper import dummy_context
from torch.utils.tensorboard import SummaryWriter


args = hp.arg_parser()
hp.print_args(args)

if args.seed != -1:
    print(f"Setting random seed to {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_determinism(seed=args.seed)

experiment_name = f"{args.name}_optim_{args.optimizer}_lr_{args.lr}_bs_{args.batch_size}_epochs_{args.epochs}_drop_{args.drop_modality}"

########### model ################
model = MOEResUNetWithBottleneck(
    in_channels=6,
    out_channels=2,
    last_layer_conv_only=True,
    num_experts=args.num_experts,
    gate_input_dim=args.context_dim,
)
teacher = None
summary(
    model,
    input_data=(torch.rand(1, 6, 128, 128, 128), torch.rand(1, args.context_dim)),
    depth=6,
    col_names=("input_size", "output_size", "num_params"),
)

################ loss and optimizer ################
dscCELoss = DiceCELoss(
    softmax=True, smooth_nr=0, include_background=False, to_onehot_y=True
)
cosSimilarityLoss = nn.CosineEmbeddingLoss()
KLdivLoss = nn.KLDivLoss(reduction="batchmean")
optimizer = hp.get_optimizer(args.optimizer, model, args.lr)
grad_scaler = GradScaler() if args.amp else None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
################# dataset ################
print("Loading datasets............")
if args.sequence == 0:
    dataset_names = ["BRATS", "ATLAS", "MSSEG", "ISLES", "WMH"]
elif args.sequence == 1:
    dataset_names = ["MSSEG", "BRATS", "ISLES", "WMH", "ATLAS"]
else:
    raise ValueError("Sequence should be either 0 or 1")

root_path = args.data_path

train_dataloader_args = {
    "num_workers": args.num_workers,
    "shuffle": True,
    "pin_memory": True,
}

test_dataloader_args = {
    "num_workers": args.num_workers,
    "shuffle": False,
    "pin_memory": True,
}

train_datasets = []
test_datasets = []
for name in dataset_names:
    train_datasets.append(
        BasicDataset(
            root_path=root_path,
            dataset_type=name,
            test_mode=False,
            randomly_drop_modalities=args.drop_modality,
            return_context=True,
            context_dim=args.context_dim,
        )
    )
    test_datasets.append(
        BasicDataset(
            root_path=root_path,
            dataset_type=name,
            test_mode=True,
            randomly_drop_modalities=False,
            return_context=True,
            context_dim=args.context_dim,
        )
    )
print("Datasets loaded!!!!!!!!!!!!")

################# continue_training ################
episode_idx = 0
epoch_idx = 0

if args.continue_training_from_episode or args.continue_training_from_epoch:
    if args.continue_training_from_episode:
        if not os.path.exists(args.load_checkpoint_path_episode):
            raise ValueError(
                f"Checkpoint path {args.load_checkpoint_path_episode} does not exist"
            )
        print(f"Loading checkpoint from {args.load_checkpoint_path_episode}")
        checkpoint = torch.load(args.load_checkpoint_path_episode, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        hp.optimizer_to(optimizer, device)
        if checkpoint["teacher"] is not None:
            teacher = deepcopy(model)
            teacher.load_state_dict(checkpoint["teacher"])
        episode_idx = checkpoint["episode_idx"]
        if checkpoint["grad_scaler"] is not None:
            grad_scaler.load_state_dict(checkpoint["grad_scaler"])
        print(f"Checkpoint for a episode {checkpoint['episode_idx']} Loaded!!!!!!")

    if args.continue_training_from_epoch:
        if not os.path.exists(args.load_checkpoint_path_epoch):
            raise ValueError(
                f"Checkpoint path {args.load_checkpoint_path_epoch} does not exist"
            )
        print(f"Loading checkpoint from {args.load_checkpoint_path_epoch}")
        checkpoint = torch.load(args.load_checkpoint_path_epoch, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        hp.optimizer_to(optimizer, device)
        if checkpoint["teacher"] is not None:
            teacher = deepcopy(model)
            teacher.load_state_dict(checkpoint["teacher"])
        epoch_idx = checkpoint["epoch"]
        if checkpoint["grad_scaler"] is not None:
            grad_scaler.load_state_dict(checkpoint["grad_scaler"])
        print(
            f"Checkpoint for a episode {checkpoint['episode_idx']} and epoch {checkpoint['epoch']} Loaded!!!!!!"
        )


################# moving to gpu ################
model = model.to(device)
if teacher is not None:
    teacher = teacher.to(device)

################# compile ################
if args.compile:
    torch._dynamo.config.traceable_tensor_subclasses.add(MetaTensor)
    print("Compiling............")
    model = torch.compile(model)
    if teacher is not None:
        teacher = torch.compile(teacher)
    # dscCELoss.dc = torch.compile(dscCELoss.dc)
    dscCELoss = torch.compile(dscCELoss)
    # explanation = torch._dynamo.explain(dscCELoss)(torch.randn(5,1,3,3,3), torch.randn(5,1,3,3,3))
    # print(explanation)
    # exit()
    cosSimilarityLoss = torch.compile(cosSimilarityLoss)
    KLdivLoss = torch.compile(KLdivLoss)
    print("Compiled!!!!!!!!!!!!")

################# tensorboard init ################
writer = SummaryWriter(os.path.join(args.checkpoint_path, experiment_name, "logs"))
################# training ################
for episode_id in range(episode_idx, len(dataset_names)):
    print(f"Training on {dataset_names[episode_id]}")
    dataset_name = dataset_names[episode_id]
    train_dataloader = DataLoader(
        train_datasets[episode_id],
        batch_size=args.batch_size,
        **train_dataloader_args,
    )
    epoch_iterator = (
        tqdm(range(epoch_idx, args.epochs))
        if args.show_progress
        else range(epoch_idx, args.epochs)
    )

    for epoch in epoch_iterator:
        start_time = time.time()
        if not args.show_progress:
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Epoch: {epoch}/{args.epochs}"
            )

        model.train()
        if teacher is not None:
            teacher.eval()
        total_loss = []
        total_dsc_loss = []
        total_cos_loss = []
        total_kl_loss = []
        for i, (img, gt, context) in enumerate(train_dataloader):
            img = img.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device.type, enabled=True) if args.amp else dummy_context():
                if episode_id == 0 or (args.alpha == 0 and args.beta == 0): # first episode or no distillation
                    pred_logits = model(img, context)
                    loss = dscCELoss(pred_logits, gt)
                    writer.add_scalar(
                        f"Loss_ep_{episode_id}/dsc_step",
                        loss.item(),
                        i + epoch * len(train_dataloader),
                    )
                    writer.add_scalar(
                        f"Loss_ep_{episode_id}/total_step",
                        loss.item(),
                        i + epoch * len(train_dataloader),
                    )
                    total_dsc_loss.append(loss.item())
                else:
                    pred_logits_main, features_main = model(
                        img, context, give_feature=True
                    )
                    with torch.no_grad():
                        pred_logits_teacher, features_teacher = teacher(
                            img, context, give_feature=True
                        )

                    cos_loss = cosSimilarityLoss(
                        features_main.reshape(gt.shape[0], -1),
                        features_teacher.reshape(gt.shape[0], -1),
                        torch.ones(features_main.shape[0]).to(device),
                    )

                    writer.add_scalar(
                        f"Loss_ep_{episode_id}/cos_step",
                        cos_loss.item(),
                        i + epoch * len(train_dataloader),
                    )
                    total_cos_loss.append(cos_loss.item())

                    kl_loss = KLdivLoss(
                        F.log_softmax(pred_logits_main / args.tempurature, dim=1),
                        F.softmax(pred_logits_teacher / args.tempurature, dim=1),
                    )
                    
                    writer.add_scalar(
                        f"Loss_ep_{episode_id}/KL_step",
                        kl_loss.item(),
                        i + epoch * len(train_dataloader),
                    )
                    total_kl_loss.append(kl_loss.item())

                    dice_loss = dscCELoss(pred_logits_main, gt)

                    writer.add_scalar(
                        f"Loss_ep_{episode_id}/dsc_step",
                        dice_loss.item(),
                        i + epoch * len(train_dataloader),
                    )
                    total_dsc_loss.append(dice_loss.item())

                    loss = dice_loss + args.beta * cos_loss + args.alpha * kl_loss
                    writer.add_scalar(
                        f"Loss_ep_{episode_id}/total_step",
                        loss.item(),
                        i + epoch * len(train_dataloader),
                    )

            if grad_scaler is None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                optimizer.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                grad_scaler.step(optimizer)
                grad_scaler.update()

            total_loss.append(loss.item())
            if i % 25 == 0 and i != 0:
                print(f"step {i}/{len(train_dataloader)-1}: {total_loss[-1]:.5f}")

        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Epoch: {epoch} Loss: {sum(total_loss)/len(total_loss):.5f}"
        )
        print(f"Time taken for epoch in seconds: {time.time() - start_time:.2f}")

        writer.add_scalar(
            f"Loss_ep_{episode_id}/total_epoch",
            sum(total_loss) / len(total_loss),
            epoch,
        )
        writer.add_scalar(
            f"Loss_ep_{episode_id}/dsc_epoch",
            sum(total_dsc_loss) / len(total_dsc_loss),
            epoch,
        )
        if not (episode_id == 0 or (args.alpha == 0 and args.beta == 0)):
            writer.add_scalar(
                f"Loss_ep_{episode_id}/cos_epoch",
                sum(total_cos_loss) / len(total_cos_loss),
                epoch,
            )
            writer.add_scalar(
                f"Loss_ep_{episode_id}/kl_epoch",
                sum(total_kl_loss) / len(total_kl_loss),
                epoch,
            )

        ################# save the latest epoch checkpoint ################

        hp.save_checkpoint(
            model,
            None,
            optimizer,
            epoch,
            episode_id,
            grad_scaler,
            args,
            experiment_name,
            save_epoch=True,
        )

    print("Training completed!!!!!!!!!!!!")
    del train_dataloader
    ############################ saving episode checkpoint ############################

    print("Saving checkpoint............")
    teacher = deepcopy(model)
    # freeze the teacher
    for param in teacher.parameters():
        param.requires_grad = False

    hp.save_checkpoint(
        model,
        None,
        optimizer,
        None,
        episode_id,
        grad_scaler,
        args,
        experiment_name,
        save_epoch=False,
    )
    print("Checkpoint saved!!!!!!!!!!!!")
    ############################ saving results ############################
    print("Evaluating............")
    results = hp.evaluate_model(
        test_datasets,
        model,
        device,
        test_dataloader_args,
        args.amp,
    )
    save_result_path = os.path.join(
        args.checkpoint_path, experiment_name, f"exprience_results_{episode_id}.json"
    )
    os.makedirs(os.path.dirname(save_result_path), exist_ok=True)
    with open(save_result_path, "w") as f:
        json.dump(results, f, indent=4)
    for key, value in results.items():
        writer.add_scalar(f"Results/{key}", value, episode_id)
    print("Evaluation completed!!!!!!!!!!!!")
    writer.flush()
    epoch_idx = 0
    ################################### changing KL divergence loss ###################################
    if (episode_id < len(dataset_names) - 1) and args.dynamic_coef:
        print("Changing KL divergence loss............")
        tmp_result = hp.tmp_evaluation(
            train_datasets[episode_id + 1],
            model,
            device,
            test_dataloader_args,  # since we just want to test on the training data to see how well we will perform on the next dataset
            args.amp,
        )
        print(f"tmp result: {tmp_result}")
        kl_coeff = 1 - tmp_result
        print(f"KL divergence coefficient: {kl_coeff}")
        args.alpha = kl_coeff * args.alpha_max * 10 ** (-5) # scaling the coefficient
        print(f"New KL divergence coefficient (args.alpha): {args.alpha}")
        writer.add_scalar(
            f"Results/KL_divergence_coefficient", kl_coeff, episode_id + 1
        )
        writer.add_scalar(
            f"Results/KL_divergence_coefficient_scaled", args.alpha, episode_id + 1
        )
        print("KL divergence loss changed!!!!!!!!!!!!")

writer.close()
