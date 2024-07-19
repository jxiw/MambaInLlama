import os
import math
import logging
from time import time
from tqdm import tqdm

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import transformers
from transformers import get_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from mamba.hybrid_wrapper import MambaTransformerHybridModelWrapper
from mamba.hybrid_mamba_config import MambaConfig

from train_configs import DistillConfig
from train_configs import DistillArgumentParser
from dataset import TextDataset

from util import load_safetensors_to_dict, construct_layer_dict

logger = get_logger(__name__)

def main():
    parser = DistillArgumentParser((DistillConfig))
    training_args = parser.parse()

    accelerator = (
        Accelerator(log_with="wandb")
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    set_seed(training_args.seed)

    if accelerator.is_main_process:
        if training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    model_name = training_args.model_name
    dtype = torch.bfloat16

    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    teacher_model = teacher_model.to(dtype)

    config = AutoConfig.from_pretrained(model_name, dtype=dtype)
    
    d_xb = config.num_key_value_heads * \
        (config.hidden_size // config.num_attention_heads)

    # d_xb = config.num_key_value_heads * config.head_dim
    
    ssm_layers = training_args.ssm_layers
    attn_layers = [i for i in range(config.num_hidden_layers) if i not in ssm_layers]
    
    mamba_config = MambaConfig(
        config.hidden_size,
        {"expand": 1},
        config.rms_norm_eps,
        d_xb=d_xb,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        n_layer=config.num_hidden_layers,
        attn_layers=attn_layers,
    )

    student_model = MambaTransformerHybridModelWrapper.init_distillation(
            None, model_name, mamba_config, attn_layers=attn_layers, init_with_kqvo=training_args.init_with_kqvo)

    if training_args.prev_checkpoint_path is not None:
        # this is for progressive distillation,
        # override ssm layers using the previous weights
        prev_checkpoint = load_safetensors_to_dict(
            training_args.prev_checkpoint_path)
        prev_checkpoint_layers, is_mamba_layer = construct_layer_dict(prev_checkpoint, config.num_hidden_layers)
        for (layer_id, layer_checkpoint) in prev_checkpoint_layers.items():
            if is_mamba_layer[layer_id]:
                # override weights of that layer
                student_model.model.model.layers[layer_id].load_state_dict(layer_checkpoint)
                
    # Freeze all parameters in teacher model by setting requires_grad to False
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Freeze all non mamba parameters in student  model
    for name, param in student_model.named_parameters():
        if f"mamba" not in name:
            param.requires_grad = False

    if accelerator.is_main_process:
        print("teacher_model:", teacher_model)
        total_params = sum(p.numel() for p in teacher_model.parameters())
        total_trainable_params = sum(
            p.numel() for p in teacher_model.parameters() if p.requires_grad)
        print(f"number of total params:{total_params}")
        print(f"number of total trainable params:{total_trainable_params}")

        print("student_model:", student_model)
        total_params = sum(p.numel() for p in student_model.parameters())
        total_trainable_params = sum(
            p.numel() for p in student_model.parameters() if p.requires_grad)
        print(f"number of total params:{total_params}")
        print(f"number of total trainable params:{total_trainable_params}")
        
        student_model.save_config(training_args.output_dir)

    # load dataset
    train_data = torch.cat([torch.load(f'{train_dataset_path}/input_ids.pt', map_location="cpu") for train_dataset_path in training_args.train_datasets_path], dim=0)
    train_label = torch.cat([torch.load(f'{train_dataset_path}/labels.pt', map_location="cpu") for train_dataset_path in training_args.train_datasets_path], dim=0)
    train_dataset = TextDataset(train_data, train_label, pad_token_id=tokenizer.pad_token_id)
    train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    print("length of dataset:", len(train_dataset))

    if training_args.do_eval:
        eval_data = torch.cat([torch.load(f'{eval_dataset_path}/input_ids.pt', map_location="cpu") for eval_dataset_path in training_args.eval_datasets_path], dim=0)
        eval_label = torch.cat([torch.load(f'{eval_dataset_path}/labels.pt', map_location="cpu") for eval_dataset_path in training_args.eval_datasets_path], dim=0)
        eval_dataset = TextDataset(eval_data, eval_label, pad_token_id=tokenizer.pad_token_id)
        eval_dataloader = DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), lr=training_args.learning_rate, betas=(0.9, 0.98))

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_steps is None or training_args.max_steps < 0:
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps,
    )

    if training_args.do_eval:
        # Prepare everything with our `accelerator`.
        student_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            student_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    else:
        # Prepare everything with our `accelerator`.
        student_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            student_model, optimizer, train_dataloader, lr_scheduler
        )

    print("length of dataloader:", len(train_dataloader))

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)

    save_steps = None
    # Figure out how many steps we should save the Accelerator states
    if training_args.save_steps is not None:
        save_steps = training_args.save_steps
    
    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if accelerator.is_main_process:
        experiment_config = vars(training_args)
        experiment_config["lr_scheduler_type"] = "cosine"
        accelerator.init_trackers("mamba_distill", experiment_config)

    # Train!
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint is not None or training_args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {training_args.resume_from_checkpoint}")
            accelerator.load_state(training_args.resume_from_checkpoint)
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    teacher_model = teacher_model.to(accelerator.device)
    teacher_model.eval()

    curr_loss = 0.0
    curr_kl_loss = 0.0
    curr_teacher_loss = 0.0
    curr_student_loss = 0.0

    # training
    for epoch in range(starting_epoch, training_args.num_train_epochs):
        start_time = time()
        student_model.train()
        
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if training_args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue

            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
                teacher_logits = teacher_outputs.logits
                teach_cross_entropy_loss = teacher_outputs.loss

            targets = F.softmax(teacher_logits, dim=-1)
            student_outputs = student_model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
            student_logits = student_outputs.logits
            student_cross_entropy_loss = student_outputs.loss

            kl_loss = F.kl_div(F.log_softmax(
                student_logits, dim=-1), targets, reduction='batchmean')

            loss = training_args.kl_weight * kl_loss + training_args.ce_weight * student_cross_entropy_loss

            curr_loss += loss.detach().float()
            curr_kl_loss += kl_loss.detach().float()
            curr_teacher_loss += teach_cross_entropy_loss.detach().float()
            curr_student_loss += student_cross_entropy_loss.detach().float()

            loss = loss / training_args.gradient_accumulation_steps
            accelerator.backward(loss)

            if (step > 0 and step % training_args.gradient_accumulation_steps == 0) or step == len(train_dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(
                    student_model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # log loss
                accelerator.print(
                    f'training loss: {curr_loss / training_args.gradient_accumulation_steps:.5f}')
                accelerator.log({'train loss': curr_loss / training_args.gradient_accumulation_steps,
                                'teacher kl loss': curr_kl_loss / training_args.gradient_accumulation_steps,
                                'teacher ce loss': curr_teacher_loss / training_args.gradient_accumulation_steps,
                                'student ce loss': curr_student_loss / training_args.gradient_accumulation_steps,
                                'lr': lr_scheduler.get_last_lr()[0], 'step': completed_steps})
                curr_loss = 0
                curr_kl_loss = 0
                curr_teacher_loss = 0
                curr_student_loss = 0
                completed_steps += 1
                progress_bar.update(1) 

            if isinstance(save_steps, int):
                if completed_steps > 0 and completed_steps % save_steps == 0:
                    accelerator.wait_for_everyone()
                    # save checkpoint
                    output_dir = f"step_{completed_steps}"
                    if training_args.output_dir is not None:
                        output_dir = os.path.join(training_args.output_dir, output_dir)
                    # save model weight
                    unwrapped_model = accelerator.unwrap_model(student_model)
                    unwrapped_model.model.save_pretrained(
                        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(output_dir)
                    accelerator.save_state(output_dir)

        end_time = time()
        logger.info(f"Epoch {epoch} training took {end_time-start_time} seconds")

        if training_args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(student_model)
            unwrapped_model.model.save_pretrained(
                training_args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(training_args.output_dir)
        
        if training_args.do_eval:
            # run evaluation
            student_model.eval()
            total_eval_loss = 0
            total_eval_kl_loss = 0
            total_eval_student_ce_loss = 0
            total_eval_teacher_ce_loss = 0
            for step, batch in enumerate(eval_dataloader):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch["attention_mask"]
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
                    teacher_logits = teacher_outputs.logits
                    teach_cross_entropy_loss = teacher_outputs.loss
                targets = F.softmax(teacher_logits, dim=-1)
                student_outputs = student_model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
                student_logits = student_outputs.logits
                student_cross_entropy_loss = student_outputs.loss
                kl_loss = F.kl_div(F.log_softmax(
                    student_logits, dim=-1), targets, reduction='batchmean')
                loss = training_args.kl_weight * kl_loss + training_args.ce_weight * student_cross_entropy_loss
                total_eval_loss += loss.detach().float()
                total_eval_kl_loss += kl_loss.detach().float()
                total_eval_student_ce_loss += student_cross_entropy_loss.detach().float()
                total_eval_teacher_ce_loss += teach_cross_entropy_loss.detach().float()

            avg_eval_loss = total_eval_loss / len(train_dataloader)
            avg_eval_kl_loss = total_eval_kl_loss / len(train_dataloader)
            avg_eval_teacher_ce_loss = total_eval_teacher_ce_loss / len(train_dataloader)
            avg_eval_student_ce_loss = total_eval_student_ce_loss / len(train_dataloader)

            avg_eval_loss = accelerator.gather(torch.tensor(avg_eval_loss).to(accelerator.device)).mean().item()
            avg_eval_kl_loss = accelerator.gather(torch.tensor(avg_eval_kl_loss).to(accelerator.device)).mean().item()
            avg_eval_teacher_ce_loss = accelerator.gather(torch.tensor(avg_eval_teacher_ce_loss).to(accelerator.device)).mean().item()
            avg_eval_student_ce_loss = accelerator.gather(torch.tensor(avg_eval_student_ce_loss).to(accelerator.device)).mean().item()

            accelerator.log({'eval loss': avg_eval_loss,
                    'eval kl loss': avg_eval_kl_loss,
                    'eval teacher ce loss': avg_eval_teacher_ce_loss,
                    'eval student ce loss': avg_eval_student_ce_loss,
                    'step': completed_steps})

if __name__ == "__main__":
    main()