import time

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score, classification_report, confusion_matrix


class CheckpointProgressCallback(TrainerCallback):
    """Prints plain-text progress toward the next epoch checkpoint (readable via `tail -f`, unlike tqdm's \\r bar)."""

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.time()
        self._start_step = state.global_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps != 0:
            return
        steps_done = state.global_step - self._start_step
        if steps_done == 0:
            return
        rate = (time.time() - self._start_time) / steps_done
        steps_per_epoch = state.max_steps / args.num_train_epochs
        epoch_progress = state.epoch - int(state.epoch)
        steps_remaining = (1 - epoch_progress) * steps_per_epoch
        print(
            f"[checkpoint progress] epoch {state.epoch:.2f}/{int(args.num_train_epochs)} "
            f"({epoch_progress * 100:.1f}% through epoch) | global step {state.global_step}/{state.max_steps} "
            f"| ~{steps_remaining * rate / 60:.1f} min to next checkpoint",
            flush=True,
        )


class Collator:
    def __call__(self, batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}


def collect_fn2(model_name):
    return Collator()


def compute_matrics_for_classification(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

class ExpertNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.network(x)

class GateNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.gate(x)

class MMOEModel(nn.Module):
    def __init__(self, base_model, num_experts=5, num_tasks=3, hidden_dim=512, class_weights=None):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.register_buffer("class_weights", class_weights, persistent=False)
        
        # Base model (Swin Transformer)
        self.base_model = base_model
        self.feature_dim = self.base_model.config.hidden_size
        
        # Expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(self.feature_dim, hidden_dim) 
            for _ in range(num_experts)
        ])
        
        # Gate networks (one per task)
        self.gates = nn.ModuleList([
            GateNetwork(self.feature_dim, num_experts)
            for _ in range(num_tasks)
        ])
        
        # Task-specific towers
        self.task_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, 2)  # Binary output per task
            ) for _ in range(num_tasks)
        ])
        
        # Final classifier for combining task outputs
        self.final_classifier = nn.Linear(num_tasks * 2, 3)  # 3 classes
        
    def forward(self, pixel_values, labels=None):
        # Get base features
        base_outputs = self.base_model(pixel_values, output_hidden_states=True)
        features = base_outputs.hidden_states[-1].mean(dim=1)  # Swin has no [CLS] token; mean-pool patches
        
        # Expert outputs
        expert_outputs = [expert(features) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, hidden_dim]
        
        # Gate outputs for each task
        gate_outputs = [gate(features) for gate in self.gates]  # List of [batch, num_experts]
        
        # Task-specific outputs
        task_outputs = []
        for task_id, gate_output in enumerate(gate_outputs):
            # Weighted combination of experts for this task
            gate_output = gate_output.unsqueeze(-1)  # [batch, num_experts, 1]
            task_input = (expert_outputs * gate_output).sum(dim=1)  # [batch, hidden_dim]
            
            # Task-specific tower
            task_output = self.task_towers[task_id](task_input)  # [batch, 2]
            task_outputs.append(task_output)
            
        # Combine task outputs
        combined_output = torch.cat(task_outputs, dim=1)  # [batch, num_tasks*2]
        logits = self.final_classifier(combined_output)  # [batch, 3]
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}

def build_classification_trainer(args, train_set, val_set, feature_extractor):
    assert args.num_classes == 3  # MMOE is designed for 3-class classification
    
    # Load base model
    base_model = AutoModelForImageClassification.from_pretrained(
        "microsoft/swin-large-patch4-window12-384-in22k",
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True
    )
    
    # Inverse-frequency class weights so the loss can't settle for always predicting the majority class
    label_counts = train_set.df['label'].value_counts().sort_index()
    class_weights = torch.tensor(
        [len(train_set.df) / (len(label_counts) * count) for count in label_counts], dtype=torch.float
    )

    # Create MMOE model
    model = MMOEModel(
        base_model=base_model,
        num_experts=5,
        num_tasks=3,
        hidden_dim=512,
        class_weights=class_weights
    )
    
    arguments = TrainingArguments(
        args.log_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=3,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        logging_steps=50,
        fp16=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        # Add weight decay and learning rate scheduling
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        # Add early stopping
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_set,
        eval_dataset=val_set,
        processing_class=feature_extractor,
        data_collator=collect_fn2("microsoft/swin-large-patch4-window12-384-in22k"),
        compute_metrics=compute_matrics_for_classification,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), CheckpointProgressCallback()],
    )
    
    return trainer, model