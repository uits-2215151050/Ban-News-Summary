# BanglaT5 Text Summarization Project

This project demonstrates how to fine-tune a BanglaT5 model for text summarization on a custom dataset using Google Colab, HuggingFace Transformers, and the `datasets` library. The model is trained to generate concise summaries from longer Bengali articles.

## Project Setup and Overview

### 1. Google Drive Integration & Cache Setup

To ensure persistence and efficient storage, the project is configured to use Google Drive for all persistent files, including model checkpoints, logs, and HuggingFace cache.

-   **Drive Mounting**: Google Drive is mounted to `/content/drive`.
-   **Directory Structure**: A base directory `BanglaT5_Project` is created in Google Drive, with subdirectories for `cache`, `checkpoints`, `model`, and `logs`.
-   **HuggingFace Cache Redirection**: `HF_HOME` and `TRANSFORMERS_CACHE` environment variables are set to the Google Drive cache directory (`/content/drive/MyDrive/BanglaT5_Project/cache`) to store downloaded models and datasets.
-   **W&B (Weights & Biases) Disabled**: W&B logging is disabled and set to offline mode to prevent external reporting.

### 2. Library Installation

Necessary Python libraries for natural language processing, dataset handling, and model training are installed:

-   `transformers`: For pre-trained models and utilities.
-   `datasets`: For efficient dataset loading and processing.
-   `sentencepiece`: Tokenizer used by T5 models.
-   `rouge-score`: For evaluating summarization models.
-   `gradio`: For potentially deploying a demo (though not used in the provided steps).

### 3. Dataset Loading and Preparation

The project uses a custom dataset consisting of Bengali articles and their corresponding summaries, stored in `article.txt` and `summary.txt` files, respectively.

-   **Loading Data**: Articles and summaries are loaded from text files into a Pandas DataFrame.
-   **Train/Validation Split**: The dataset is split into training and validation sets with a 90/10 ratio using `sklearn.model_selection.train_test_split`.
-   **HuggingFace Dataset Conversion**: The Pandas DataFrames are converted into `datasets.Dataset` objects for compatibility with the HuggingFace `Trainer`.

### 4. Model and Tokenizer Loading

The `csebuetnlp/banglat5` model and its corresponding tokenizer are loaded from HuggingFace Hub.

-   **Tokenizer**: `AutoTokenizer.from_pretrained("csebuetnlp/banglat5")` is used.
-   **Model**: `AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/banglat5")` is used.
-   **Cache**: Both the tokenizer and model are loaded using the specified `cache_dir` in Google Drive.

### 5. Data Preprocessing / Tokenization

A `preprocess` function is defined to prepare the text data for the T5 model.

-   **Input Formatting**: Articles are prefixed with "summarize: " as required by T5 models for summarization tasks.
-   **Tokenization**: Both input articles and target summaries are tokenized using the loaded tokenizer, with `max_length` set to 128 for input and 32 for output, and `truncation` and `padding` enabled.
-   **Label Handling**: For sequence-to-sequence models, padding tokens in labels are replaced with -100, which are ignored by the loss function.
-   **Batch Processing**: The `preprocess` function is applied to the training and validation datasets using `map` with `batched=True`.

### 6. Data Collator Setup

A `DataCollatorForSeq2Seq` is used to dynamically pad input sequences and labels to the longest sequence in each batch during training.

-   `tokenizer`: The same tokenizer used for preprocessing.
-   `model`: The loaded T5 model.
-   `padding=True`: Enables dynamic padding.
-   `label_pad_token_id=-100`: Ensures padding tokens in labels are ignored during loss calculation.

### 7. Training Arguments

`TrainingArguments` are configured to specify the training parameters and logging/saving behavior.

-   `output_dir`: Checkpoints are saved to `f"{BASE_DIR}/checkpoints"`.
-   `per_device_train_batch_size` & `per_device_eval_batch_size`: Set to 4.
-   `gradient_accumulation_steps`: Set to 2.
-   `learning_rate`: `3e-4`.
-   `num_train_epochs`: 5 epochs.
-   `save_strategy` & `eval_strategy`: Both set to `"epoch"`.
-   `logging_dir`: Logs are saved to `f"{BASE_DIR}/logs"`.
-   `logging_steps`: Logs every 100 steps.
-   `load_best_model_at_end`: The best model based on `eval_loss` is loaded after training.
-   `metric_for_best_model`: `"eval_loss"`.
-   `report_to`: Set to `"none"` to disable reporting to external services.

### 8. Trainer Setup

The `Trainer` class from HuggingFace Transformers is initialized with the model, training arguments, datasets, data collator, and tokenizer.

### 9. Model Training and Saving

-   **Training**: The `trainer.train()` method is called to start the fine-tuning process.
-   **Saving**: After training, the final trained model and tokenizer are saved to `f"{BASE_DIR}/model"` in Google Drive.

### 10. Test Summary Generation

A `summarize` function is defined to test the trained model with new text.

-   **Input**: Takes a Bengali article as input.
-   **Preprocessing**: Formats the input with "summarize: " and tokenizes it.
-   **Generation**: Uses `model.generate()` to create a summary, with `max_length=32` and `num_beams=4`.
-   **Decoding**: Decodes the generated token IDs back to human-readable text, skipping special tokens.
-   **Example**: Demonstrates summarization using an article from the validation set, comparing the generated summary with the true summary.
