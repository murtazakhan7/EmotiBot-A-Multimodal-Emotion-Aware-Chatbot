"""
EmotiBot - Session 2: Emotion-Aware Response Generation
Fine-tuning Flan-T5 with LoRA on Empathetic Dialogue Datasets

This notebook implements Phase 2 of the EmotiBot project:
- Load EmpatheticDialogues, ESConv, and DailyDialog datasets
- Integrate emotion labels from Session 1 model
- Fine-tune Flan-T5-base with LoRA for empathetic response generation
- Evaluate and save the model

Environment: Google Colab (Free Tier)
"""

# ============================================================================
# SETUP AND INSTALLATIONS
# ============================================================================

# Install required packages
!pip install -q transformers datasets accelerate peft bitsandbytes evaluate rouge_score

# Import libraries
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import evaluate
from google.colab import drive
import os
import json
from datetime import datetime
from tqdm.auto import tqdm

# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Use existing project directory
PROJECT_DIR = '/content/drive/MyDrive/EmotiBot'
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
PLOTS_DIR = os.path.join(PROJECT_DIR, 'plots')
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')
DATA_DIR = os.path.join(PROJECT_DIR, 'processed_data')

os.makedirs(DATA_DIR, exist_ok=True)

print(f"✓ Using project directory: {PROJECT_DIR}")

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# LOAD SESSION 1 EMOTION CLASSIFIER
# ============================================================================

print("\n" + "="*80)
print("LOADING SESSION 1 EMOTION CLASSIFIER")
print("="*80)

from transformers import pipeline

# Load the emotion classifier from Session 1
emotion_model_path = os.path.join(MODELS_DIR, 'distilbert_emotion_classifier')

try:
    emotion_classifier = pipeline(
        "text-classification",
        model=emotion_model_path,
        device=0 if torch.cuda.is_available() else -1,
        top_k=None
    )
    print(f"✓ Loaded emotion classifier from: {emotion_model_path}")
except Exception as e:
    print(f"⚠ Could not load Session 1 model: {e}")
    print("  Using emotion labels from dataset instead")
    emotion_classifier = None

# Load emotion labels
with open(os.path.join(MODELS_DIR, 'emotion_labels.json'), 'r') as f:
    emotion_mapping = json.load(f)

print(f"✓ Loaded {len(emotion_mapping['id2label'])} emotion labels")

# ============================================================================
# LOAD AND PREPARE DIALOGUE DATASETS
# ============================================================================

print("\n" + "="*80)
print("LOADING DIALOGUE DATASETS")
print("="*80)

# 1. Load EmpatheticDialogues
print("\n1. Loading EmpatheticDialogues...")
empathetic_dialogues = load_dataset("empathetic_dialogues")

print(f"   Train: {len(empathetic_dialogues['train'])}")
print(f"   Validation: {len(empathetic_dialogues['validation'])}")
print(f"   Test: {len(empathetic_dialogues['test'])}")

# 2. Load DailyDialog
print("\n2. Loading DailyDialog...")
daily_dialog = load_dataset("daily_dialog")

print(f"   Train: {len(daily_dialog['train'])}")
print(f"   Validation: {len(daily_dialog['validation'])}")
print(f"   Test: {len(daily_dialog['test'])}")

# 3. ESConv (Emotional Support Conversation)
print("\n3. Loading ESConv...")
try:
    esconv = load_dataset("thu-coai/esconv")
    print(f"   Train: {len(esconv['train'])}")
except Exception as e:
    print(f"   ⚠ Could not load ESConv: {e}")
    print("   Continuing with EmpatheticDialogues and DailyDialog only")
    esconv = None

# ============================================================================
# PROCESS EMPATHETIC DIALOGUES
# ============================================================================

print("\n" + "="*80)
print("PROCESSING EMPATHETIC DIALOGUES")
print("="*80)

def process_empathetic_dialogues(dataset_split):
    """Convert EmpatheticDialogues to input-output format"""
    
    processed = []
    
    for example in tqdm(dataset_split, desc="Processing"):
        emotion = example['context']  # The emotion/situation
        prompt = example['prompt']  # User utterance
        response = example['utterances'][-1] if example['utterances'] else ""
        
        if prompt and response:
            processed.append({
                'emotion': emotion,
                'input_text': prompt,
                'target_text': response
            })
    
    return Dataset.from_dict({
        'emotion': [x['emotion'] for x in processed],
        'input_text': [x['input_text'] for x in processed],
        'target_text': [x['target_text'] for x in processed]
    })

empathetic_train = process_empathetic_dialogues(empathetic_dialogues['train'])
empathetic_val = process_empathetic_dialogues(empathetic_dialogues['validation'])

print(f"✓ Processed EmpatheticDialogues:")
print(f"  Train: {len(empathetic_train)}")
print(f"  Validation: {len(empathetic_val)}")

# ============================================================================
# PROCESS DAILY DIALOG
# ============================================================================

print("\n" + "="*80)
print("PROCESSING DAILY DIALOG")
print("="*80)

# DailyDialog emotion mapping
DAILY_DIALOG_EMOTIONS = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise"
}

def process_daily_dialog(dataset_split):
    """Convert DailyDialog to input-output format"""
    
    processed = []
    
    for example in tqdm(dataset_split, desc="Processing"):
        dialog = example['dialog']
        emotions = example['emotion']
        
        # Create turn-by-turn conversations
        for i in range(len(dialog) - 1):
            emotion_id = emotions[i] if i < len(emotions) else 0
            emotion = DAILY_DIALOG_EMOTIONS.get(emotion_id, "neutral")
            
            processed.append({
                'emotion': emotion,
                'input_text': dialog[i],
                'target_text': dialog[i + 1]
            })
    
    return Dataset.from_dict({
        'emotion': [x['emotion'] for x in processed],
        'input_text': [x['input_text'] for x in processed],
        'target_text': [x['target_text'] for x in processed]
    })

daily_train = process_daily_dialog(daily_dialog['train'])
daily_val = process_daily_dialog(daily_dialog['validation'])

print(f"✓ Processed DailyDialog:")
print(f"  Train: {len(daily_train)}")
print(f"  Validation: {len(daily_val)}")

# ============================================================================
# MERGE DATASETS
# ============================================================================

print("\n" + "="*80)
print("MERGING DATASETS")
print("="*80)

# Combine training sets
train_dataset = concatenate_datasets([empathetic_train, daily_train])
val_dataset = concatenate_datasets([empathetic_val, daily_val])

# Shuffle
train_dataset = train_dataset.shuffle(seed=42)
val_dataset = val_dataset.shuffle(seed=42)

print(f"✓ Combined datasets:")
print(f"  Train: {len(train_dataset)}")
print(f"  Validation: {len(val_dataset)}")

# Sample examples
print("\n--- Sample Training Examples ---")
for i in range(3):
    example = train_dataset[i]
    print(f"\nExample {i+1}:")
    print(f"  Emotion: {example['emotion']}")
    print(f"  Input: {example['input_text'][:100]}...")
    print(f"  Target: {example['target_text'][:100]}...")

# ============================================================================
# PREPARE DATA FOR FLAN-T5
# ============================================================================

print("\n" + "="*80)
print("PREPARING DATA FOR FLAN-T5")
print("="*80)

# Initialize Flan-T5 tokenizer
MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"✓ Loaded tokenizer: {MODEL_NAME}")

# Create prompt template
def create_prompt(emotion, text):
    """Create emotion-aware prompt for Flan-T5"""
    return f"Respond with {emotion} to: {text}"

def preprocess_function(examples):
    """Tokenize inputs and targets"""
    
    # Create prompts with emotion context
    inputs = [
        create_prompt(emotion, text) 
        for emotion, text in zip(examples['emotion'], examples['input_text'])
    ]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True,
        padding=False  # Dynamic padding
    )
    
    # Tokenize targets
    labels = tokenizer(
        examples['target_text'],
        max_length=128,
        truncation=True,
        padding=False
    )
    
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs

# Process datasets
print("Tokenizing datasets...")
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train"
)

tokenized_val = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing validation"
)

print(f"✓ Tokenization complete!")

# ============================================================================
# INITIALIZE FLAN-T5 WITH LORA
# ============================================================================

print("\n" + "="*80)
print("INITIALIZING FLAN-T5 WITH LORA")
print("="*80)

# Load base model
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

print(f"✓ Loaded base model: {MODEL_NAME}")

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    target_modules=["q", "v"],  # Apply LoRA to attention layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

# Apply LoRA to model
model = get_peft_model(base_model, lora_config)

print("✓ LoRA applied to model")

# Print trainable parameters
model.print_trainable_parameters()

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("TRAINING CONFIGURATION")
print("="*80)

# Metrics
rouge = evaluate.load('rouge')

def compute_metrics(eval_pred):
    """Compute ROUGE scores"""
    
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels (used for padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    return {
        'rouge1': result['rouge1'],
        'rouge2': result['rouge2'],
        'rougeL': result['rougeL']
    }

# Data collator for dynamic padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Training arguments (optimized for Free Colab)
training_args = TrainingArguments(
    output_dir=os.path.join(MODELS_DIR, 'flan_t5_lora_checkpoints'),
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    learning_rate=1e-4,
    per_device_train_batch_size=4,  # Small batch for memory
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir=LOGS_DIR,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True,
    dataloader_num_workers=2,
    predict_with_generate=True,
    generation_max_length=128,
)

print("Training Arguments:")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Epochs: {training_args.num_train_epochs}")

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("✓ Trainer initialized")

# ============================================================================
# MODEL TRAINING
# ============================================================================

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)

start_time = datetime.now()
print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print("⚠ This may take 4-6 hours on Free Colab. Stay connected!")

train_result = trainer.train()

end_time = datetime.now()
training_duration = end_time - start_time

print(f"\n✓ Training completed!")
print(f"  Duration: {training_duration}")
print(f"  Final loss: {train_result.training_loss:.4f}")

# ============================================================================
# MODEL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# Evaluate on validation set
print("Evaluating on validation set...")
val_metrics = trainer.evaluate()

print("\nValidation Metrics:")
for key, value in val_metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")

# Save metrics
metrics_summary = {
    'model': MODEL_NAME,
    'lora_config': {
        'r': lora_config.r,
        'lora_alpha': lora_config.lora_alpha,
        'target_modules': lora_config.target_modules
    },
    'training_duration': str(training_duration),
    'validation_metrics': {k: float(v) for k, v in val_metrics.items() if isinstance(v, float)},
    'training_args': {
        'learning_rate': training_args.learning_rate,
        'batch_size': training_args.per_device_train_batch_size,
        'epochs': training_args.num_train_epochs
    }
}

with open(os.path.join(LOGS_DIR, 'session2_metrics.json'), 'w') as f:
    json.dump(metrics_summary, f, indent=2)

print(f"\n✓ Metrics saved to {LOGS_DIR}")

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

# Save LoRA weights
lora_save_path = os.path.join(MODELS_DIR, 'flan_t5_lora_emotion_response')
model.save_pretrained(lora_save_path)
tokenizer.save_pretrained(lora_save_path)

print(f"✓ LoRA weights saved to: {lora_save_path}")

# Also save config for easy loading
config_info = {
    'base_model': MODEL_NAME,
    'lora_path': lora_save_path,
    'emotion_classifier_path': emotion_model_path
}

with open(os.path.join(MODELS_DIR, 'session2_config.json'), 'w') as f:
    json.dump(config_info, f, indent=2)

# ============================================================================
# INFERENCE TESTING
# ============================================================================

print("\n" + "="*80)
print("TESTING INFERENCE")
print("="*80)

def generate_response(emotion, user_text, max_length=100):
    """Generate empathetic response"""
    
    # Create prompt
    prompt = create_prompt(emotion, user_text)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Test examples
test_cases = [
    ("happiness", "I just got accepted into my dream university!"),
    ("sadness", "I lost my pet dog today and I'm heartbroken."),
    ("anger", "My coworker took credit for my work in the meeting."),
    ("fear", "I have a big presentation tomorrow and I'm so nervous."),
    ("surprise", "I can't believe I won the lottery!")
]

print("\nTesting model on sample inputs:\n")
for emotion, text in test_cases:
    response = generate_response(emotion, text)
    print(f"Emotion: {emotion}")
    print(f"User: {text}")
    print(f"Bot: {response}")
    print("-" * 60)

# ============================================================================
# VISUALIZE TRAINING HISTORY
# ============================================================================

print("\n" + "="*80)
print("VISUALIZING TRAINING HISTORY")
print("="*80)

# Extract training history
history = trainer.state.log_history

# Separate train and eval logs
train_logs = [log for log in history if 'loss' in log and 'eval_loss' not in log]
eval_logs = [log for log in history if 'eval_loss' in log]

if train_logs and eval_logs:
    # Plot training loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    train_steps = [log['step'] for log in train_logs]
    train_loss = [log['loss'] for log in train_logs]
    eval_steps = [log['step'] for log in eval_logs]
    eval_loss = [log['eval_loss'] for log in eval_logs]
    
    axes[0].plot(train_steps, train_loss, label='Train Loss', linewidth=2)
    axes[0].plot(eval_steps, eval_loss, label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ROUGE-L plot
    eval_rougeL = [log['eval_rougeL'] for log in eval_logs]
    
    axes[1].plot(eval_steps, eval_rougeL, label='ROUGE-L', 
                color='green', linewidth=2)
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('ROUGE-L Score')
    axes[1].set_title('Validation ROUGE-L Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'session2_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Training history plot saved to: {plot_path}")

# ============================================================================
# SESSION 2 SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SESSION 2 COMPLETE ✓")
print("="*80)

summary = f"""
SESSION 2 SUMMARY - Emotion-Aware Response Generation
------------------------------------------------------
Model: Flan-T5-base + LoRA
Datasets: EmpatheticDialogues + DailyDialog
Training samples: {len(train_dataset)}

LoRA Configuration:
- Rank (r): {lora_config.r}
- Alpha: {lora_config.lora_alpha}
- Target modules: {lora_config.target_modules}
- Trainable parameters: ~{sum(p.numel() for p in model.parameters() if p.requires_grad):,}

Performance:
- Validation ROUGE-L: {val_metrics['eval_rougeL']:.4f}
- Validation ROUGE-1: {val_metrics['eval_rouge1']:.4f}
- Validation ROUGE-2: {val_metrics['eval_rouge2']:.4f}
- Training time: {training_duration}

Saved Artifacts:
✓ LoRA weights: {lora_save_path}
✓ Metrics: {LOGS_DIR}/session2_metrics.json
✓ Config: {MODELS_DIR}/session2_config.json
✓ Plots: {PLOTS_DIR}/session2_training_history.png

Next Steps (Session 3):
→ Fine-tune ViT on FER2013 for image emotion detection
→ Implement multimodal fusion
→ Create end-to-end inference pipeline
"""

print(summary)

# Save summary
with open(os.path.join(LOGS_DIR, 'session2_summary.txt'), 'w') as f:
    f.write(summary)

print(f"\n✓ Summary saved to {LOGS_DIR}")
print("\nReady to proceed to Session 3! 🚀")