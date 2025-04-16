BERT-based News Classification
This project uses BERT (Bidirectional Encoder Representations from Transformers) for classifying news articles into different categories based on their descriptions.

Dataset Columns: Class Index, Title, and Description. We use the Description for training.

Label Encoding: The Class Index is encoded into numerical labels using LabelEncoder.

Tokenizer: bert-base-uncased tokenizer is used to convert text into input IDs and attention masks.

Model: BertForSequenceClassification with a classification head is fine-tuned for multi-class classification.

Training: The model learns to predict the correct category using cross-entropy loss and the AdamW optimizer.
