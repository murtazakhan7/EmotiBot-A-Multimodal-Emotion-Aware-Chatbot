# 🤖 EmotiBot - Multimodal Emotion-Aware Chatbot



A state-of-the-art multimodal chatbot that detects emotions from both text and facial expressions, then generates empathetic, context-aware responses.

## 🌟 Key Features

- **Multimodal Emotion Detection**: Analyzes both text and facial expressions
- **27 Text Emotions**: Fine-grained emotion classification using DistilBERT
- **7 Facial Emotions**: Facial expression recognition using Vision Transformer
- **Empathetic Responses**: Context-aware reply generation with Flan-T5 + LoRA
- **Safety First**: Built-in toxicity filtering with Detoxify
- **Interactive Demo**: Easy-to-use Gradio interface

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [Usage Examples](#usage-examples)
- [Model Details](#model-details)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Citation](#citation)

## 🏗️ Architecture

```
User Input (Text + Optional Image)
           ↓
    ┌──────┴──────┐
    ↓             ↓
Text Emotion   Image Emotion
(DistilBERT)   (ViT)
    ↓             ↓
    └──────┬──────┘
           ↓
   Multimodal Fusion
           ↓
   Flan-T5 + LoRA
  (Response Gen)
           ↓
   Detoxify Filter
           ↓
  Empathetic Response
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Google Colab account (for training)
- Kaggle API token (for FER2013 dataset)

### Setup

1. **Clone or download the project notebooks**

2. **Install dependencies** (automatically handled in notebooks):
```bash
pip install transformers datasets accelerate peft pillow detoxify gradio
```

3. **Mount Google Drive** (for model storage):
```python
from google.colab import drive
drive.mount('/content/drive')
```

## ⚡ Quick Start

### Option 1: Use Pre-trained Models (If Available)

```python
# Load the complete EmotiBot pipeline
from emotibot import emotibot_pipeline

# Text-only inference
result = emotibot_pipeline(
    user_text="I'm so excited about my new project!",
    user_image=None
)

print(result['response'])
# Output: "That's wonderful! Your excitement is contagious..."

# Multimodal inference (text + image)
from PIL import Image
image = Image.open('happy_face.jpg')

result = emotibot_pipeline(
    user_text="How do I look today?",
    user_image=image
)
```

### Option 2: Train From Scratch

Follow the 4-session training pipeline:

1. **Session 1**: Text Emotion Detection (~1-2 hours)
2. **Session 2**: Response Generation (~4-6 hours)
3. **Session 3**: Image Emotion Detection (~2-3 hours)
4. **Session 4**: Complete Integration (~30 minutes)

## 📚 Training Pipeline

### Session 1: Text Emotion Detection

**Goal**: Fine-tune DistilBERT on GoEmotions for 27-class emotion classification

**Dataset**: [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (58K Reddit comments)

**Key Steps**:
```python
# 1. Load dataset
dataset = load_dataset("go_emotions", "simplified")

# 2. Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=27,
    problem_type="multi_label_classification"
)

# 3. Train with mixed precision
training_args = TrainingArguments(
    per_device_train_batch_size=16,
    num_train_epochs=3,
    fp16=True
)

# 4. Save model
trainer.save_model("distilbert_emotion_classifier")
```

**Output**: 
- Trained model: `distilbert_emotion_classifier/`
- Validation F1: ~0.50-0.55
- Training time: ~1-2 hours on Free Colab

### Session 2: Response Generation

**Goal**: Fine-tune Flan-T5 with LoRA for emotion-aware response generation

**Datasets**: 
- [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues) (25K conversations)
- [DailyDialog](https://huggingface.co/datasets/daily_dialog) (13K dialogues)
- ESConv (Optional)

**Key Steps**:
```python
# 1. Merge datasets
train_dataset = concatenate_datasets([
    process_empathetic_dialogues(empathetic_train),
    process_daily_dialog(daily_train)
])

# 2. Apply LoRA to Flan-T5
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(base_model, lora_config)

# 3. Fine-tune with gradient accumulation
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch: 16
    num_train_epochs=3
)

# 4. Save LoRA weights
model.save_pretrained("flan_t5_lora_emotion_response")
```

**Output**:
- LoRA weights: `flan_t5_lora_emotion_response/`
- Validation ROUGE-L: ~0.30-0.40
- Training time: ~4-6 hours on Free Colab

### Session 3: Image Emotion Detection

**Goal**: Fine-tune ViT on FER2013 for facial emotion recognition

**Dataset**: [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) (35K facial images)

**Setup Kaggle**:
```python
# 1. Upload kaggle.json
from google.colab import files
uploaded = files.upload()

# 2. Download dataset
!kaggle datasets download -d msambare/fer2013 -p ./data --unzip
```

**Two-Stage Training**:
```python
# Stage 1: Train classification head only (2 epochs)
for param in model.vit.parameters():
    param.requires_grad = False

# Stage 2: Fine-tune entire model (3 epochs)
for param in model.parameters():
    param.requires_grad = True
```

**Output**:
- Trained model: `vit_emotion_classifier/`
- Validation F1: ~0.55-0.65
- Training time: ~2-3 hours on Free Colab

### Session 4: Integration & Demo

**Goal**: Integrate all models and create interactive demo

**Components**:
1. Load all three models
2. Implement multimodal fusion
3. Add safety filtering (Detoxify)
4. Create Gradio interface
5. Benchmark performance

**Usage**:
```python
# Launch interactive demo
demo.launch(share=True)
# Returns public URL for remote access
```

## 💡 Usage Examples

### Basic Text Interaction

```python
# Happy emotion
result = emotibot_pipeline(
    user_text="I just got promoted! I'm so happy!",
    user_image=None
)
print(result['emotion_analysis']['fused_emotion']['primary_emotion'])
# Output: "joy"

print(result['response'])
# Output: "Congratulations on your promotion! That's fantastic news..."
```

### Multimodal Interaction

```python
# Sad face + sad text
sad_image = Image.open('sad_face.jpg')
result = emotibot_pipeline(
    user_text="I'm feeling really down today.",
    user_image=sad_image
)

# Both modalities detected sadness → higher confidence
print(result['emotion_analysis']['fused_emotion'])
# Output: {
#   'primary_emotion': 'sadness',
#   'confidence': 0.87,
#   'source': 'fused_aligned'
# }
```

### API Response Structure

```python
{
    'user_input': {
        'text': str,
        'has_image': bool
    },
    'emotion_analysis': {
        'text_emotions': [
            {'emotion': 'joy', 'confidence': 0.85},
            {'emotion': 'excitement', 'confidence': 0.72}
        ],
        'image_emotion': {
            'emotion': 'happy',
            'confidence': 0.91
        },
        'fused_emotion': {
            'primary_emotion': 'joy',
            'confidence': 0.88,
            'source': 'fused_aligned'
        }
    },
    'response': "That's wonderful! Your happiness is truly inspiring...",
    'safety': {
        'is_safe': True,
        'toxicity_scores': {...}
    },
    'metadata': {
        'processing_time': 0.45,
        'timestamp': '2025-10-21T...'
    }
}
```

## 🎯 Model Details

| Component | Model | Parameters | Dataset | Metric |
|-----------|-------|------------|---------|--------|
| Text Emotion | DistilBERT-base | 67M | GoEmotions (58K) | F1: 0.50-0.55 |
| Image Emotion | ViT-base | 86M | FER2013 (35K) | F1: 0.55-0.65 |
| Response Gen | Flan-T5-base + LoRA | 250M (2M trainable) | EmpatheticDialogues + DailyDialog | ROUGE-L: 0.30-0.40 |
| Safety Filter | Detoxify | - | RealToxicityPrompts | - |

### Emotion Categories

**Text Emotions (27)**:
admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise

**Facial Emotions (7)**:
angry, disgust, fear, happy, neutral, sad, surprise

## 📊 Performance

### Inference Speed (Free Colab T4 GPU)

| Mode | Average Time | Std Dev |
|------|--------------|---------|
| Text Only | 0.35s | ±0.08s |
| Text + Image | 0.52s | ±0.12s |

### Accuracy

- **Text Emotion Classification**: Multi-label F1 ~0.50
- **Image Emotion Classification**: F1 ~0.60  
- **Response Quality**: ROUGE-L ~0.35

### Memory Requirements

- **Training**: ~12GB GPU RAM (fits Free Colab)
- **Inference**: ~4GB GPU RAM

## 📁 Project Structure

```
EmotiBot/
├── Session1_TextEmotion.ipynb          # Text emotion training
├── Session2_ResponseGen.ipynb          # Response generation training
├── Session3_ImageEmotion.ipynb         # Image emotion training
├── Session4_Integration.ipynb          # Complete integration
├── README.md                           # This file
├── USAGE_GUIDE.md                      # Detailed usage instructions
├── ARCHITECTURE.txt                    # System architecture
├── models/
│   ├── distilbert_emotion_classifier/  # Text emotion model
│   ├── flan_t5_lora_emotion_response/  # Response generator
│   ├── vit_emotion_classifier/         # Image emotion model
│   └── emotion_labels.json             # Label mappings
├── logs/
│   ├── session1_metrics.json
│   ├── session2_metrics.json
│   ├── session3_metrics.json
│   ├── session4_summary.txt
│   └── pipeline_benchmark.json
├── plots/
│   ├── emotion_distribution_*.png
│   ├── fer2013_confusion_matrix.png
│   ├── session*_training_history.png
│   └── fer2013_predictions.png
└── processed_data/
    └── fer2013_data/                   # Downloaded FER2013 dataset
```

## 🎓 Implementation Details

### Hybrid Architecture Rationale

**Why not just use Flan-T5 end-to-end?**

The hybrid approach (DistilBERT → Flan-T5) offers several advantages:

1. **Explicit Emotion Detection**: Provides interpretable emotion labels
2. **Modular Debugging**: Each component can be tested independently  
3. **Better Control**: Explicit emotions guide response generation
4. **Multi-task Learning**: Specialized models for specialized tasks

### LoRA Benefits

Using LoRA (Low-Rank Adaptation) for Flan-T5 fine-tuning:

- **Memory Efficient**: Only 2M trainable parameters vs 250M
- **Faster Training**: ~50% reduction in training time
- **No Catastrophic Forgetting**: Preserves base model capabilities
- **Easy Deployment**: Can merge or keep separate LoRA weights

### Two-Stage ViT Training

**Stage 1: Warm-up (2 epochs)**
- Freeze base ViT layers
- Train only classification head
- Higher learning rate (5e-4)

**Stage 2: Fine-tune (3 epochs)**
- Unfreeze all layers
- Lower learning rate (2e-5)
- Refines feature extraction

This prevents overfitting and improves convergence.

## 🔧 Configuration & Hyperparameters

### Session 1: DistilBERT

```python
{
    "model": "distilbert-base-uncased",
    "learning_rate": 2e-5,
    "batch_size": 16,
    "epochs": 3,
    "max_length": 128,
    "fp16": True,
    "warmup_ratio": 0.1
}
```

### Session 2: Flan-T5 + LoRA

```python
{
    "model": "google/flan-t5-base",
    "lora_r": 8,
    "lora_alpha": 32,
    "learning_rate": 1e-4,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "epochs": 3,
    "max_length": 128
}
```

### Session 3: ViT

```python
{
    "model": "google/vit-base-patch16-224-in21k",
    "stage1_lr": 5e-4,
    "stage2_lr": 2e-5,
    "batch_size": 16,
    "stage1_epochs": 2,
    "stage2_epochs": 3,
    "image_size": 224
}
```

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory (OOM) Error**
```python
# Solution: Reduce batch size
per_device_train_batch_size=8  # Instead of 16
gradient_accumulation_steps=2   # To maintain effective batch size
```

**2. Kaggle Dataset Download Fails**
```python
# Ensure kaggle.json is uploaded correctly
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

**3. Session Timeout in Colab**
```python
# Enable background execution
# Runtime → Change runtime type → Background execution: ON
# Or periodically save checkpoints
```

**4. Model Loading Error**
```python
# Verify paths exist
import os
assert os.path.exists('/content/drive/MyDrive/EmotiBot/models/')
```

**5. Gradio Not Launching**
```python
# Try without sharing
demo.launch(share=False)

# Or update gradio
!pip install --upgrade gradio
```

## 🚦 Best Practices

### For Training

1. **Use Checkpointing**: Models save automatically during training
2. **Monitor GPU Memory**: Use `nvidia-smi` to check usage
3. **Save Frequently**: Google Colab can disconnect
4. **Use Mixed Precision**: `fp16=True` for faster training
5. **Early Stopping**: Prevents overfitting

### For Inference

1. **Batch Processing**: Process multiple inputs together
2. **Cache Models**: Load once, use many times
3. **GPU Inference**: 5-10x faster than CPU
4. **Input Validation**: Check text length and image size
5. **Error Handling**: Graceful fallbacks for edge cases

## 🔒 Safety & Ethics

### Built-in Safety Features

1. **Toxicity Filtering**: Detoxify checks all responses
2. **Fallback Responses**: Safe defaults for toxic content
3. **Emotion Validation**: Confidence thresholds prevent misclassification

### Ethical Considerations

⚠️ **Important Limitations**:

- Not a replacement for mental health professionals
- May misinterpret sarcasm or complex emotions
- Cultural differences in emotional expression
- Privacy concerns with facial image processing
- Potential biases from training data

### Responsible Use

✅ **Good Use Cases**:
- Educational demonstrations
- Research on emotion recognition
- Customer service applications
- Entertainment chatbots

❌ **Inappropriate Use Cases**:
- Medical diagnosis or treatment
- Legal or official assessments
- Surveillance without consent
- High-stakes decision making

## 📈 Future Enhancements

### Short-term (Community Contributions Welcome!)

- [ ] Multi-language support (translate prompts)
- [ ] Conversation history/context tracking
- [ ] Batch inference optimization
- [ ] Model quantization (INT8) for faster inference
- [ ] Docker containerization

### Long-term

- [ ] Voice input/output integration
- [ ] Real-time video emotion tracking
- [ ] Multi-speaker conversation handling
- [ ] Personalized response adaptation
- [ ] Production deployment (FastAPI + Redis)
- [ ] Fine-tuning on domain-specific data

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Better Datasets**: Suggest or contribute new empathetic dialogue data
2. **Optimization**: Speed up inference, reduce memory usage
3. **Evaluation**: More comprehensive metrics and benchmarks
4. **Documentation**: Improve guides, add tutorials
5. **Features**: Implement items from Future Enhancements

## 📄 License

This project is released under the MIT License. See `LICENSE` file for details.

**Note**: Individual components use different licenses:
- Transformers: Apache 2.0
- Datasets: Various (check individual dataset licenses)
- Pre-trained models: Typically Apache 2.0 or MIT

## 📚 Citation

If you use EmotiBot in your research, please cite:

```bibtex
@software{emotibot2025,
  title={EmotiBot: A Multimodal Emotion-Aware Chatbot},
  author={Muhammad Murtaza},
  year={2025},
  url={https://github.com/murtazakhan7/EmotiBot-A-Multimodal-Emotion-Aware-Chatbot}
}
```

### Datasets Used

**GoEmotions**:
```bibtex
@inproceedings{demszky2020goemotions,
  title={GoEmotions: A Dataset of Fine-Grained Emotions},
  author={Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
  booktitle={ACL},
  year={2020}
}
```

**EmpatheticDialogues**:
```bibtex
@inproceedings{rashkin2019towards,
  title={Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset},
  author={Rashkin, Hannah and Smith, Eric Michael and Li, Margaret and Boureau, Y-Lan},
  booktitle={ACL},
  year={2019}
}
```

**FER2013**:
```bibtex
@article{goodfellow2013challenges,
  title={Challenges in representation learning: A report on three machine learning contests},
  author={Goodfellow, Ian J and Erhan, Dumitru and Carrier, Pierre Luc and Courville, Aaron and Mirza, Mehdi and Hamner, Ben and Cukierski, Will and Tang, Yichuan and Thaler, David and Lee, Dong-Hyun and others},
  journal={Neural Networks},
  year={2013}
}
```

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library
- **Google Research** for GoEmotions dataset
- **Facebook Research** for EmpatheticDialogues
- **Kaggle** for hosting FER2013
- **Anthropic** for Claude (assisted in project planning)

## 📞 Support

For questions, issues, or suggestions:

- **Email**: murtazamuhammad508@gmail.com
- **Documentation**: See `developerguide` for detailed instructions

## 🌐 Resources

### Official Documentation

- [Transformers Docs](https://huggingface.co/docs/transformers)
- [PEFT (LoRA) Guide](https://huggingface.co/docs/peft)
- [Gradio Documentation](https://gradio.app/docs)

### Tutorials & Papers

- [Fine-tuning Language Models](https://huggingface.co/course)
- [Vision Transformers Explained](https://arxiv.org/abs/2010.11929)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

### Related Projects

- [TextEmotionClassification](https://github.com/example/text-emotion)
- [FacialEmotionRecognition](https://github.com/example/fer)
- [EmpathicChatbots](https://github.com/example/empathic-chatbot)

---

## 🎉 Quick Start Checklist

Ready to run EmotiBot? Follow this checklist:

### Pre-Training Setup
- [ ] Google Colab account created
- [ ] Google Drive mounted in Colab
- [ ] Kaggle account created (for FER2013)
- [ ] kaggle.json downloaded
- [ ] GPU runtime enabled in Colab

### Session 1: Text Emotion (~1-2 hours)
- [ ] Run all cells in Session 1 notebook
- [ ] Model saved to Drive: `distilbert_emotion_classifier/`
- [ ] Validation F1 > 0.45
- [ ] Test inference working

### Session 2: Response Generation (~4-6 hours)
- [ ] Run all cells in Session 2 notebook
- [ ] LoRA weights saved: `flan_t5_lora_emotion_response/`
- [ ] ROUGE-L > 0.25
- [ ] Sample responses generated successfully

### Session 3: Image Emotion (~2-3 hours)
- [ ] kaggle.json uploaded
- [ ] FER2013 downloaded successfully
- [ ] Stage 1 training complete
- [ ] Stage 2 training complete
- [ ] Model saved: `vit_emotion_classifier/`
- [ ] F1 > 0.50

### Session 4: Integration (~30 min)
- [ ] All three models loaded successfully
- [ ] Pipeline test cases pass
- [ ] Gradio demo launches
- [ ] Public URL generated
- [ ] Documentation generated

### Testing
- [ ] Text-only inference works
- [ ] Image-only inference works
- [ ] Multimodal inference works
- [ ] Safety filter active
- [ ] No OOM errors

**If all items are checked, EmotiBot is ready! 🚀**

---

**Built with ❤️ by the EmotiBot Team**

*Last Updated: October 2025*
