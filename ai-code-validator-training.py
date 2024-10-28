# AI/ML Trained Python Code Validator with Web Interface

# Code: Aleksi Bovellan
# Gradio interface: Satu Pohjonen


# AI/ML MODEL TRAINING SCRIPT FOR TRAINING A NEW INTELLIGENCE FOLDER ("RESULTS")


# Set the "sample_size" below to your needs - it determines the complexity of the training database and the time that training will take.
# Example 1: sample_size of 1200 takes about 10-15 minutes to train on a year 2020 MacBook M1 Air
# Example 2: sample_size of 30000 takes around 8,5 hours to train on the same machine

sample_size = 30000  # Adjust this value for faster training or better model performance

# From here on, the training will run automatically. Finally, it will evaluate the confidence in learned results.
# This script also prints out progress reports for debugging purposes along the way.


# Import necessary libraries
import random
import re
import numpy as np
import warnings
import sys
from transformers import (
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaTokenizer,
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from datasets import load_dataset, Dataset, concatenate_datasets

# Load the dataset and tokenizer
print("\n")
print("Loading dataset...")
dataset = load_dataset("code_search_net", "python")
print("Loading tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Function to remove unnecessary comments and docstrings from code examples which are correct
def remove_comments_and_docstrings(code):
    """
    Remove comments and docstrings from the given code string.
    """
    # Remove docstrings (both triple double and triple single quotes)
    code = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', '', code)
    # Remove single-line comments
    code = re.sub(r'#.*', '', code)
    return code

# Function to limit code examlpe length to a maximum number of lines
def limit_code_length(code, max_lines=20):
    """
    Limit the code to a maximum number of lines.
    """
    lines = code.strip().split('\n')
    return '\n'.join(lines[:max_lines])

# Function to check if code examples are syntactically correct
def is_syntax_correct(code):
    """
    Check if the given code string is syntactically correct.
    Suppress syntax warnings to prevent cluttering the output.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SyntaxWarning)
            compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False

# Corruption functions to introduce syntax errors for examples of wrong Python code to learn from
def remove_closing_parenthesis_or_quote(code):
    """
    Remove a closing parenthesis or quote if present at the end of the code.
    """
    if code and code[-1] in [')', '"', "'"]:
        code = code[:-1]
    return code

def remove_opening_parenthesis_or_quote(code):
    """
    Remove an opening parenthesis or quote if present at the start of the code.
    """
    if code and code[0] in ['(', '"', "'"]:
        code = code[1:]
    return code

def remove_colon_after_def(code):
    """
    Remove the colon after a function definition.
    """
    pattern = r'(def .*\))\s*:'
    matches = list(re.finditer(pattern, code))
    if matches:
        match = random.choice(matches)
        start, end = match.span()
        code = code[:end-1] + code[end:]  # Remove the colon
    return code

def remove_parenthesis_in_def(code):
    """
    Remove parentheses in a function definition.
    """
    pattern = r'def\s+\w+\(.*\):'
    if re.match(pattern, code):
        code = re.sub(r'\(|\)', '', code)
    return code

def remove_quotes_in_string(code):
    """
    Remove quotes in string literals.
    """
    code = re.sub(r'(["\'])(.*?)(["\'])', r'\2', code)
    return code

# Function to actually corrupt code by applying those corruption functions
def corrupt_code(code):
    """
    Corrupt the code by applying one of the corruption functions.
    """
    corruptions = [
        remove_closing_parenthesis_or_quote,
        remove_opening_parenthesis_or_quote,
        remove_colon_after_def,
        remove_parenthesis_in_def,
        remove_quotes_in_string,
    ]
    corrupted_code = code
    attempts = 0
    max_attempts = 5
    while attempts < max_attempts:
        corruption = random.choice(corruptions)
        corrupted_code = corruption(code)
        # Ensure corrupted code is syntactically incorrect
        if not is_syntax_correct(corrupted_code) and corrupted_code != code:
            return corrupted_code
        attempts += 1
    # If unable to corrupt, return original code
    return corrupted_code

# Function to compare tokenizations of original and corrupted code examples
def compare_tokenizations(code, corrupted_code):
    """
    Compare tokenizations of the original and corrupted code.
    """
    tokens_original = tokenizer.tokenize(code)
    tokens_corrupted = tokenizer.tokenize(corrupted_code)
    print("Tokens are different:", tokens_original != tokens_corrupted)
    print()

# Function to create augmented datasets with corrupted code
def create_augmented_dataset(split_dataset, dataset_name):
    """
    Create an augmented dataset by adding corrupted versions of the code.
    """
    code_list = []
    label_list = []
    print(f"\nCreating augmented dataset for {dataset_name}...")
    for idx, example in enumerate(split_dataset):
        code = example['func_code_string']
        if code and code.strip():
            # Preprocess code
            code = remove_comments_and_docstrings(code)
            code = limit_code_length(code)
            if code.strip():
                # Add correct example
                code_list.append(code)
                label_list.append(1)
                # Create corrupted code
                corrupted_code = corrupt_code(code)
                code_list.append(corrupted_code)
                label_list.append(0)  # Corrupted code is incorrect
                # Debugging: Print examples for the first few entries
                if idx < 3:
                    print(f"--- Example {idx + 1} ---")
                    print("Original code:")
                    print(code[:200] + "\n")
                    print("Corrupted code:")
                    print(corrupted_code[:200] + "\n")
                    # Compare tokenizations
                    compare_tokenizations(code, corrupted_code)
    data_dict = {'func_code_string': code_list, 'label': label_list}
    return Dataset.from_dict(data_dict)

# Function to also create some manual examples of correct and incorrect codes
def create_test_examples_dataset():
    """
    Create a dataset with test examples to include in the training dataset.
    """
    test_examples = [
        ('print("hello")', 1),
        ('print("hello"', 0),
        ('print("hello', 0),
        ('print("hello)', 0),
        ('print(hello', 0),
        ('def provide_feedback(code):', 1),
        ('def provide_feedback(code)', 0),
        ('def provide_feedback(code:', 0),
        ('def provide_feedback(code', 0),
        ('''
from datasets import load_dataset

# Load Python dataset from CodeSearchNet
dataset = load_dataset("code_search_net", "python")

# Print out the dataset structure
print(dataset)
''', 1),
        # Introduce errors in the longer code snippet
        ('''
from datasets import load_dataset

# Load Python dataset from CodeSearchNet
dataset = load_dataset("code_search_net", "python"

# Print out the dataset structure
print(dataset)
''', 0),
        ('for i in range(10):\n    print(i)', 1),
        ('for i in range(10)\n    print(i)', 0),
        ('if x == 5:\n    print("x is 5")', 1),
        ('if x == 5\n    print("x is 5")', 0),
    ]
    codes, labels = zip(*test_examples)
    data_dict = {'func_code_string': codes, 'label': labels}
    return Dataset.from_dict(data_dict)

# Main execution wrapped in a try-except block for graceful exit
try:
    # Prepare datasets
    print("Preparing datasets...")
    # Shuffle and select a subset of the original dataset
    original_dataset = dataset['train'].shuffle(seed=42).select(range(sample_size))

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(original_dataset))
    valid_size = len(original_dataset) - train_size

    train_dataset_original = original_dataset.select(range(train_size))
    valid_dataset_original = original_dataset.select(range(train_size, train_size + valid_size))

    # Create augmented training and validation datasets
    train_dataset = create_augmented_dataset(train_dataset_original, "training")
    valid_dataset = create_augmented_dataset(valid_dataset_original, "validation")

    # Add manual test examples to the training dataset
    test_examples_dataset = create_test_examples_dataset()
    train_dataset = concatenate_datasets([train_dataset, test_examples_dataset])

    # Re-shuffle the training dataset
    train_dataset = train_dataset.shuffle(seed=42)

    # Check label distribution
    train_labels = train_dataset['label']
    valid_labels = valid_dataset['label']
    print(f"\nTraining set label distribution: {np.bincount(train_labels)}")
    print(f"Validation set label distribution: {np.bincount(valid_labels)}")

    # Tokenization function including labels
    def preprocess_function(examples):
        """
        Tokenize the code examples and assign labels.
        """
        result = tokenizer(examples['func_code_string'], truncation=True, padding='max_length', max_length=256)
        result['labels'] = examples['label']
        return result

    # Tokenize datasets
    print("\nTokenizing training dataset...")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['func_code_string', 'label'])
    print("Tokenizing validation dataset...")
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True, remove_columns=['func_code_string', 'label'])

    # Ensure labels are correctly assigned
    print(f"\nTokenized training set labels: {tokenized_train_dataset['labels'][:10]}")
    print(f"Tokenized validation set labels: {tokenized_valid_dataset['labels'][:10]}")

    # Define the model
    print("\nLoading model...")
    model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  # Updated parameter name
        save_strategy="no",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_dir='./logs',
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=False,
    )

    # Define metrics for evaluation
    def compute_metrics(eval_pred):
        """
        Compute evaluation metrics.
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = np.mean(predictions == labels)
        return {'accuracy': accuracy}

    # Initialize Trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("\nStarting training...")
    trainer.train()

    # Evaluate the model
    print("\nEvaluating model...")
    eval_result = trainer.evaluate()
    print(f"Evaluation results: {eval_result}")

    # Save the final model and tokenizer
    print("\nSaving model and tokenizer...")
    trainer.save_model("./results")
    tokenizer.save_pretrained("./results")

    # Load the trained model and create a pipeline for evaluation
    print("\nLoading trained model for evaluation...")
    classifier = pipeline("text-classification", model="./results", tokenizer="./results", framework="pt")

    # Define code snippets for evaluation
    code_snippets = [
        ('print("hello")', 'Correct Code'),
        ('print("hello"', 'Incorrect Code'),
        ('def add(a, b):\n    return a + b', 'Correct Code'),
        ('def add(a, b)\n    return a + b', 'Incorrect Code'),
        # Add more code snippets as needed
    ]

    # Evaluate and print the classification results
    print("\nClassification Results:")
    for idx, (code_snippet, description) in enumerate(code_snippets, 1):
        result = classifier(code_snippet)
        label = result[0]['label']
        score = result[0]['score']
        print(f"{idx}. {description}: {label} (Confidence: {score:.2f})")
        
    # Proceed with a bit more of model confidence analysis
    # Load the fine-tuned model and tokenizer from the 'results' folder
    model_analysis = AutoModelForSequenceClassification.from_pretrained("./results")
    tokenizer_analysis = AutoTokenizer.from_pretrained("./results")

    # Create a text classification pipeline
    classifier_analysis = pipeline("text-classification", model=model_analysis, tokenizer=tokenizer_analysis, device=-1)

    # Example input: You can replace this with any Python code snippet you'd like to evaluate
    code_snippet_1_analysis = "def add(a, b):\n    return a + b" # Correct example
    code_snippet_2_analysis = "def add(a, b)\n    return a + b"  # Incorrect, missing colon

    # Classify the code snippets
    result_1_analysis = classifier_analysis(code_snippet_1_analysis)
    result_2_analysis = classifier_analysis(code_snippet_2_analysis)

    # Print the classification results for both snippets
    print("\n")
    print(f"Classification result for code_snippet_1: {result_1_analysis}")
    print(f"Classification result for code_snippet_2: {result_2_analysis}")
    
    # Inform about the end of processing
    print("\n")
    print("**************")
    print("\n")
    print("If no errors were printed: training and evaluation was completed successfully!")
    print("You can now proceed to launch the Gradio interface script and use it with your web browser :)")
    print("\n")

# General polishing with some graceful exists
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Exiting gracefully...")
    sys.exit(0)
except Exception as e:
    print(f"\nAn error occurred: {e}")
    sys.exit(1)