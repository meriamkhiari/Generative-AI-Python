import tkinter as tk
from tkinter import scrolledtext
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag, ne_chunk
from transformers import pipeline, set_seed
import random

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Set seed for reproducibility
set_seed(random.randint(1, 10000))

def preprocess_text(text):
    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    porter = PorterStemmer()
    tokens = [porter.stem(token) for token in tokens]
    
    return tokens

def perform_sentiment_analysis(text):
    # Perform sentiment analysis using Vader
    sentiment_score = sia.polarity_scores(text)
    
    return sentiment_score

def generate_text(text):
    # Initialize a text generation pipeline using GPT-2 model
    text_generator = pipeline("text-generation", model="gpt2")
    
    # Generate text based on the input sentence
    generated_response = text_generator(text, max_length=100, num_return_sequences=1)[0]['generated_text']
    
    return generated_response

def process_input():
    # Get user input from GUI
    sentence = user_input.get("1.0", tk.END).strip()

    if sentence.lower() == 'exit':
        output_box.insert(tk.END, "Exiting program...\n")
        return

    # Clear previous outputs
    output_box.delete("1.0", tk.END)

    # Preprocess the sentence
    preprocessed_tokens = preprocess_text(sentence)
    
    # Perform sentiment analysis
    sentiment_score = perform_sentiment_analysis(sentence)

    # Generate text based on the input sentence
    generated_response = generate_text(sentence)

    # Display results in the output box
    output_box.insert(tk.END, f"Original Sentence: {sentence}\n\n")
    output_box.insert(tk.END, "Sentiment: ")
    
    # Determine overall sentiment
    if sentiment_score['compound'] >= 0.05:
        output_box.insert(tk.END, "Positive\n\n", 'positive')
    elif sentiment_score['compound'] <= -0.05:
        output_box.insert(tk.END, "Negative\n\n", 'negative')
    else:
        output_box.insert(tk.END, "Neutral\n\n", 'neutral')

    # Display generated response
    output_box.insert(tk.END, "Generated Response:\n")
    output_box.insert(tk.END, generated_response + "\n\n")

    # Print NER tags to VS Code terminal
    print("Named Entities:", ne_chunk(pos_tag(preprocess_text(sentence))))

def main():
    # Initialize GUI window
    root = tk.Tk()
    root.title("NLP Analysis Tool")

    # Set window size and position
    window_width = 800
    window_height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    root.geometry(f'{window_width}x{window_height}+{x}+{y}')

    # Define custom colors
    bg_color = '#f0f0f0'  # Light gray
    button_color = '#4CAF50'  # Green
    text_color = '#333333'  # Dark gray
    positive_color = '#90EE90'  # Light green
    neutral_color = '#FFFFE0'  # Light yellow
    negative_color = '#FFCCCB'  # Light coral

    root.configure(bg=bg_color)

    # Create input box
    input_frame = tk.Frame(root, bg=bg_color)
    input_frame.pack(pady=20)

    input_label = tk.Label(input_frame, text="Enter a sentence:", bg=bg_color, fg=text_color, font=('Arial', 14))
    input_label.pack(side=tk.LEFT, padx=10)

    global user_input
    user_input = tk.Text(input_frame, height=3, width=50, font=('Arial', 12))
    user_input.pack(side=tk.LEFT, padx=10)

    # Create button to trigger analysis
    analyze_button = tk.Button(root, text="Analyze", command=process_input, bg=button_color, fg='white', font=('Arial', 12, 'bold'))
    analyze_button.pack(pady=10)

    # Create scrolled text box for output
    global output_box
    output_box = scrolledtext.ScrolledText(root, height=20, width=80, font=('Arial', 12))
    output_box.pack(padx=20, pady=20)
    output_box.tag_configure('positive', background=positive_color, font=('Arial', 12, 'bold'))
    output_box.tag_configure('neutral', background=neutral_color, font=('Arial', 12, 'bold'))
    output_box.tag_configure('negative', background=negative_color, font=('Arial', 12, 'bold'))

    # Run the GUI
    root.mainloop()

if __name__ == "__main__":
    main()
