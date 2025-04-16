import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import NewsClassifier
from dataset import NewsDataset
from train import train_model

def main():
    # Load and preprocess data
    df = pd.read_json('data1.json', lines=True)
    df['text'] = df['headline'] + ' ' + df['short_description']
    
    # Convert categories to numerical labels
    categories = df['category'].unique()
    category_to_id = {cat: idx for idx, cat in enumerate(categories)}
    df['label'] = df['category'].map(category_to_id)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].values, 
        df['label'].values,
        test_size=0.2,
        random_state=42
    )
    
    # Create datasets
    train_dataset = NewsDataset(train_texts, train_labels)
    val_dataset = NewsDataset(val_texts, val_labels)
    
    # Initialize model
    model = NewsClassifier(num_classes=len(categories))
    
    # Train model
    trained_model = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=5,
        batch_size=32
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()