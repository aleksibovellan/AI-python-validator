# AI/ML Trained Python Code Validator with Web Interface

# Code: Aleksi Bovellan (2024)
# Gradio Web Interface: Satu Pohjonen (2024)


# AI/ML MODEL AND MAIN GRADIO WEB INTERFACE SCRIPT


# This script runs the actual AI/ML model, and also runs the localhost main Gradio web interface.
# It needs a "results" folder in the same directory, which has the training intelligence data.
# This script also prints out progress reports for debugging purposes along its bootup process.


import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import sys

# Function Definitions
#######################

def provide_feedback(code):
    """
    Provide AI-based feedback suggestions using detailed rule-based checks.
    Analyzes the code and returns a list of feedback strings.
    """
    feedback = []
    lines = code.splitlines()

    for line_num, line in enumerate(lines, start=1):
        # Check for missing return statement in functions
        if 'def ' in line and 'return' not in line:
            feedback.append(f"Line {line_num}: Your function might be missing a return statement.")
        
        # Check for unclosed parentheses
        if '(' in line and ')' not in line:
            feedback.append(f"Line {line_num}: You might have an unclosed parenthesis.")
        
        # Check for unclosed quotation marks
        if "'" in line or '"' in line:
            if line.count("'") % 2 != 0:
                feedback.append(f"Line {line_num}: You might have unclosed single quotation marks.")
            if line.count('"') % 2 != 0:
                feedback.append(f"Line {line_num}: You might have unclosed double quotation marks.")
        
        # Check for indentation issues
        if '    ' not in line and 'def ' in line and not line.startswith('def '):
            feedback.append(f"Line {line_num}: You might need to add proper indentation for this function definition.")
    
    return feedback

def analyze_code(code_example):
    """
    Analyze the user's input code.
    Returns a string with the analysis result and feedback.
    """
    print("\n")
    print("Analyzing code input...")
    # Tokenize the user's input code
    inputs = tokenizer(code_example, return_tensors="pt", truncation=True, padding=True)

    # Perform prediction (get logits)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits (raw predictions)
    logits = outputs.logits

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)

    # Separate probabilities for classes
    incorrect_prob = probs[0][0].item() * 100  # First class (e.g., incorrect)
    correct_prob = probs[0][1].item() * 100    # Second class (e.g., correct)

    # Provide AI-based feedback
    feedback = provide_feedback(code_example)
    
    # Create result message
    if correct_prob > incorrect_prob:
        result = f"Your code is {correct_prob:.2f}% likely to be correct."
    else:
        result = f"Your code is {incorrect_prob:.2f}% likely to be incorrect."

    # Add suggestions to the result message
    if feedback:
        result += "\n\nSuggested Improvements:\n" + "\n".join(f"- {item}" for item in feedback)
    else:
        result += "\n\nThe code seems to be well-written; no obvious issues detected."
    
    return result

def analyze_code_from_file(file):
    """
    Load and analyze code from a file uploaded by the user.
    Returns the analysis result.
    """
    print("Analyzing code from file...")
    try:
        # Open the file and read the content
        with open(file.name, "r", encoding="utf-8") as f:
            code = f.read()  # Read file content as text
        return analyze_code(code)  # Analyze the code content
    except Exception as e:
        return f"Error processing the file: {str(e)}"

def clear_inputs():
    """
    Clear the input fields in the Gradio interface.
    """
    return "", None

def create_interface():
    """
    Create and return the Gradio interface.
    """
    print("Setting up Gradio interface...")
    # Gradio interface with line numbers, file upload support, and clearing functionality
    with gr.Blocks() as interface:
        gr.Markdown("# AI/ML Trained Python Code Validator with Web Interface")
    
        with gr.Row():
            # Code input area with line numbers (using the 'Code' component for better readability)
            code_input = gr.Code(language="python", lines=20, label="Input your code here (Line numbers enabled)")
            
            # Option to upload code from a file
            code_file_input = gr.File(label="Or upload a code file")
    
        # Output area for analysis result
        output = gr.Textbox(lines=10, label="Analysis Result")
    
        with gr.Row():
            # Button for manual code input analysis
            submit_button = gr.Button("Analyze Code")
            submit_button.click(analyze_code, inputs=code_input, outputs=output)
    
            # Button for code analysis from file
            file_submit_button = gr.Button("Analyze Code from File")
            file_submit_button.click(analyze_code_from_file, inputs=code_file_input, outputs=output)
    
            # Button to clear both input fields (manual and file upload)
            clear_button = gr.Button("Clear")
            clear_button.click(clear_inputs, outputs=[code_input, code_file_input])
    
    return interface

# Main Execution
################

def main():
    """
    Main function to run the application.
    """
    print("\n")
    print("Starting Code Analyzer application...")

    # Declare global variables before assigning to them
    global tokenizer, model

    # Load the fine-tuned model from the checkpoint directory
    print("Loading tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModelForSequenceClassification.from_pretrained("./results/")
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        sys.exit(1)

    # Create the Gradio interface
    interface = create_interface()

    # Launch the Gradio application
    print("Launching Gradio web interface...")
    print("\n")
    try:
        interface.launch()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user. Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred during interface launch: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user. Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)
