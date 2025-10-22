"""
EmotiBot - Session 1: Text Emotion Detection
Fine-tuning DistilBERT on GoEmotions Dataset

This notebook implements Phase 1 of the EmotiBot project:
- Load and preprocess GoEmotions dataset
- Fine-tune DistilBERT for 27-class emotion classification
- Evaluate and save the model to Google Drive

Environment: Google Colab (Free Tier)
"""

# ============================================================================
# SETUP AND INSTALLATIONS
# ============================================================================

# Install required packages
!pip install -q transformers datasets accelerate evaluate scikit-learn

# Import libraries
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import evaluate
from google.colab import drive
import os
import json
from datetime import datetime

# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Create project directory in Drive
PROJECT_DIR = '/content/drive/MyDrive/EmotiBot'
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
PLOTS_DIR = os.path.join(PROJECT_DIR, 'plots')
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')

for directory in [PROJECT_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

print(f"✓ Project directory created at: {PROJECT_DIR}")

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# LOAD AND EXPLORE GOEMOTIONS DATASET
# ============================================================================

print("\n" + "="*80)
print("LOADING GOEMOTIONS DATASET")
print("="*80)

# Load GoEmotions from Hugging Face
dataset = load_dataset("go_emotions", "simplified")

print(f"\n✓ Dataset loaded successfully!")
print(f"  Train samples: {len(dataset['train'])}")
print(f"  Validation samples: {len(dataset['validation'])}")
print(f"  Test samples: {len(dataset['test'])}")

# Explore dataset structure
print("\n--- Sample Data ---")
print(dataset['train'][0])

# Get emotion labels
emotion_labels = dataset['train'].features['labels'].feature.names
num_labels = len(emotion_labels)
print(f"\n✓ Number of emotion classes: {num_labels}")
print(f"  Emotions: {emotion_labels}")

# Create label mapping
id2label = {i: label for i, label in enumerate(emotion_labels)}
label2id = {label: i for i, label in enumerate(emotion_labels)}

# Save label mapping
with open(os.path.join(MODELS_DIR, 'emotion_labels.json'), 'w') as f:
    json.dump({'id2label': id2label, 'label2id': label2id}, f, indent=2)

print(f"✓ Label mapping saved to {MODELS_DIR}")

# ============================================================================
# DATA ANALYSIS AND VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("DATASET ANALYSIS")
print("="*80)

def analyze_dataset_distribution(dataset_split, split_name):
    """Analyze and visualize emotion distribution"""
    
    # Count emotion occurrences (GoEmotions has multi-label)
    emotion_counts = {label: 0 for label in emotion_labels}
    
    for example in dataset_split:
        for label_idx in example['labels']:
            emotion_counts[emotion_labels[label_idx]] += 1
    
    # Create DataFrame for visualization
    df = pd.DataFrame(list(emotion_counts.items()), 
                      columns=['Emotion', 'Count'])
    df = df.sort_values('Count', ascending=False)
    
    # Plot distribution
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x='Emotion', y='Count', palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Emotion Distribution - {split_name} Set', fontsize=14, fontweight='bold')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, f'emotion_distribution_{split_name.lower()}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ {split_name} set distribution:")
    print(df.to_string(index=False))
    
    return df

# Analyze train set
train_dist = analyze_dataset_distribution(dataset['train'], 'Train')

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

# Initialize tokenizer
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"✓ Loaded tokenizer: {MODEL_NAME}")

def preprocess_function(examples):
    """Tokenize text and prepare labels for multi-label classification"""
    
    # Tokenize text
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128  # Reduced for memory efficiency
    )
    
    # Convert multi-label to binary vector
    labels = []
    for label_list in examples['labels']:
        label_vector = [0.0] * num_labels
        for label_idx in label_list:
            label_vector[label_idx] = 1.0
        labels.append(label_vector)
    
    tokenized['labels'] = labels
    return tokenized

# Apply preprocessing
print("Tokenizing datasets...")
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset['train'].column_names
)

print("✓ Tokenization complete!")
print(f"  Train: {len(tokenized_datasets['train'])} samples")
print(f"  Validation: {len(tokenized_datasets['validation'])} samples")
print(f"  Test: {len(tokenized_datasets['test'])} samples")

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

print("\n" + "="*80)
print("MODEL INITIALIZATION")
print("="*80)

# Load pre-trained DistilBERT for multi-label classification
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id
)

# Move model to device
model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✓ Model loaded: {MODEL_NAME}")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size: ~{total_params * 4 / 1e6:.2f} MB")

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("TRAINING CONFIGURATION")
print("="*80)

# Define metrics
def compute_metrics(eval_pred):
    """Compute metrics for multi-label classification"""
    
    predictions, labels = eval_pred
    
    # Apply sigmoid to get probabilities
    probs = 1 / (1 + np.exp(-predictions))
    
    # Convert to binary predictions (threshold = 0.5)
    preds = (probs > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = (preds == labels).all(axis=1).mean()
    
    # Per-class metrics
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for i in range(num_labels):
        tp = ((preds[:, i] == 1) & (labels[:, i] == 1)).sum()
        fp = ((preds[:, i] == 1) & (labels[:, i] == 0)).sum()
        fn = ((preds[:, i] == 0) & (labels[:, i] == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    
    return {
        'accuracy': accuracy,
        'macro_precision': np.mean(precision_per_class),
        'macro_recall': np.mean(recall_per_class),
        'macro_f1': np.mean(f1_per_class)
    }

# Training arguments (optimized for Free Colab)
training_args = TrainingArguments(
    output_dir=os.path.join(MODELS_DIR, 'distilbert_checkpoints'),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Adjusted for free Colab
    per_device_eval_batch_size=32,
    num_train_epochs=3,  # Limited for time constraint
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir=LOGS_DIR,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=2,  # Keep only 2 best checkpoints
    fp16=True,  # Mixed precision for faster training
    dataloader_num_workers=2,
    remove_unused_columns=False,
)

print("Training Arguments:")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Mixed precision (fp16): {training_args.fp16}")

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("✓ Trainer initialized with early stopping")

# ============================================================================
# MODEL TRAINING
# ============================================================================

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)

# Train model
start_time = datetime.now()
print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

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
val_metrics = trainer.evaluate(tokenized_datasets['validation'])

print("\nValidation Metrics:")
for key, value in val_metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")

# Evaluate on test set
print("\nEvaluating on test set...")
test_metrics = trainer.evaluate(tokenized_datasets['test'])

print("\nTest Metrics:")
for key, value in test_metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")

# Get predictions for detailed analysis
predictions = trainer.predict(tokenized_datasets['test'])
probs = 1 / (1 + np.exp(-predictions.predictions))
preds = (probs > 0.5).astype(int)

# Save detailed metrics
metrics_summary = {
    'model': MODEL_NAME,
    'training_duration': str(training_duration),
    'validation_metrics': {k: float(v) for k, v in val_metrics.items() if isinstance(v, float)},
    'test_metrics': {k: float(v) for k, v in test_metrics.items() if isinstance(v, float)},
    'training_args': {
        'learning_rate': training_args.learning_rate,
        'batch_size': training_args.per_device_train_batch_size,
        'epochs': training_args.num_train_epochs
    }
}

with open(os.path.join(LOGS_DIR, 'session1_metrics.json'), 'w') as f:
    json.dump(metrics_summary, f, indent=2)

print(f"\n✓ Metrics saved to {LOGS_DIR}")

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

# Save model and tokenizer
model_save_path = os.path.join(MODELS_DIR, 'distilbert_emotion_classifier')
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"✓ Model saved to: {model_save_path}")

# ============================================================================
# INFERENCE TESTING
# ============================================================================

print("\n" + "="*80)
print("TESTING INFERENCE")
print("="*80)

def predict_emotion(text, threshold=0.3):
    """Predict emotions for input text"""
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                      max_length=128, padding=True).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    
    # Get emotions above threshold
    detected_emotions = []
    for idx, prob in enumerate(probs):
        if prob > threshold:
            detected_emotions.append({
                'emotion': emotion_labels[idx],
                'confidence': float(prob)
            })
    
    # Sort by confidence
    detected_emotions = sorted(detected_emotions, 
                              key=lambda x: x['confidence'], 
                              reverse=True)
    
    return detected_emotions

# Test examples
test_texts = [
    "I'm so happy and excited about my new job!",
    "I feel really sad and lonely today.",
    "This is absolutely disgusting and makes me angry!",
    "I'm afraid something bad is going to happen.",
    "I love spending time with my family."
]

print("\nTesting model on sample texts:\n")
for text in test_texts:
    emotions = predict_emotion(text)
    print(f"Text: '{text}'")
    print(f"Detected emotions:")
    for em in emotions[:3]:  # Show top 3
        print(f"  - {em['emotion']}: {em['confidence']:.3f}")
    print()

# ============================================================================
# SESSION 1 SUMMARY
# ============================================================================

print("="*80)
print("SESSION 1 COMPLETE ✓")
print("="*80)

summary = f"""
SESSION 1 SUMMARY - Text Emotion Detection
-------------------------------------------
Model: DistilBERT-base-uncased
Dataset: GoEmotions (27 emotions)
Training samples: {len(dataset['train'])}

Performance:
- Validation F1: {val_metrics['eval_macro_f1']:.4f}
- Test F1: {test_metrics['eval_macro_f1']:.4f}
- Training time: {training_duration}

Saved Artifacts:
✓ Model: {model_save_path}
✓ Metrics: {LOGS_DIR}/session1_metrics.json
✓ Plots: {PLOTS_DIR}

Next Steps (Session 2):
→ Prepare datasets for Flan-T5 fine-tuning
→ Implement LoRA for efficient fine-tuning
→ Train emotion-aware response generator
"""

print(summary)

# Save summary
with open(os.path.join(LOGS_DIR, 'session1_summary.txt'), 'w') as f:
    f.write(summary)

print(f"\n✓ Summary saved to {LOGS_DIR}")
print("\nReady to proceed to Session 2! 🚀")