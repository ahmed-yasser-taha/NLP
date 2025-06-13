import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import string
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import tensorflow as tf
import nltk
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from datetime import datetime

# Download stopwords (run once)
nltk.download('stopwords')
arabic_stopwords = stopwords.words('arabic')
punctuations = [punc for punc in string.punctuation]

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in arabic_stopwords]
    return ' '.join(words)

# Prediction function
def predict_review(review):
    global model, vectorizer
    cleaned_review = preprocess_text(review)
    review_seq = vectorizer([cleaned_review])
    prediction = model.predict(review_seq)[0][0]
    sentiment = 'positive' if prediction > 0.5 else 'negative'
    return sentiment, prediction

# Load and process Excel data
def load_and_process():
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
    if file_path:
        try:
            df = pd.read_excel(file_path)
            if 'review_description' not in df.columns or 'rating' not in df.columns:
                raise ValueError("Excel file must contain 'review_description' and 'rating' columns.")
            
            df['cleaned_text'] = df['review_description'].apply(preprocess_text)
            label_encoder = LabelEncoder()
            df['encoded_rating'] = label_encoder.fit_transform(df['rating'])
            df['prediction'], df['confidence'] = zip(*df['cleaned_text'].apply(lambda x: predict_review(x)))
            
            update_table(df)
            update_chart(df)
            status_label.config(text="✅ Excel data loaded and predictions generated!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process file: {str(e)}")

# Update table with predictions
def update_table(df):
    tree.delete(*tree.get_children())
    for index, row in df[['review_description', 'rating', 'prediction', 'confidence']].iterrows():
        tree.insert("", "end", values=(row['review_description'], row['rating'], row['prediction'], f"{row['confidence']:.2f}"))

# Update pie chart
def update_chart(df):
    if not hasattr(df, 'prediction'):
        return
        
    # Clear previous chart
    for widget in chart_frame.winfo_children():
        widget.destroy()
    
    # Create new chart
    fig, ax = plt.subplots(figsize=(5, 3), dpi=80, facecolor='#2c2c2c')
    ax.set_facecolor('#2c2c2c')
    
    # Count sentiments
    counts = df['prediction'].value_counts()
    labels = counts.index.tolist()
    sizes = counts.values.tolist()
    colors = ['#2ecc71', '#e74c3c']  # Green for positive, red for negative
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, textprops={'color': 'white'}, 
        wedgeprops={'edgecolor': '#1a1a1a', 'linewidth': 1}
    )
    
    ax.set_title('Sentiment Distribution', color='white', pad=20)
    
    # Add chart to Tkinter
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Predict and save to Excel
def predict_manual():
    review = entry.get()
    if review:
        try:
            sentiment, confidence = predict_review(review)
            result_label.config(text=f"🌟 {sentiment} (Confidence: {confidence:.2f})")
            
            # Save to Excel
            new_data = pd.DataFrame({
                'review': [review],
                'predicted_sentiment': [sentiment],
                'confidence': [confidence],
                'timestamp': [pd.Timestamp.now()]
            })
            excel_file = 'comment_history.xlsx'
            if os.path.exists(excel_file):
                existing_df = pd.read_excel(excel_file)
                updated_df = pd.concat([existing_df, new_data], ignore_index=True)
            else:
                updated_df = new_data
            updated_df.to_excel(excel_file, index=False)
            status_label.config(text=f"✅ Prediction saved to '{excel_file}'!")
            
            # Clear entry field after prediction
            entry.delete(0, tk.END)
        except Exception as e:
            result_label.config(text=f"⚠️ Error: {str(e)}")
    else:
        result_label.config(text="⚠️ Please enter a review first!")

# Copy selected text
def copy_text():
    selected_item = tree.focus()
    if selected_item:
        item_data = tree.item(selected_item)
        review_text = item_data['values'][0]
        root.clipboard_clear()
        root.clipboard_append(review_text)
        status_label.config(text="✅ Text copied to clipboard!")

# Cut selected text
def cut_text():
    copy_text()
    selected_item = tree.focus()
    if selected_item:
        tree.delete(selected_item)
        status_label.config(text="✅ Text cut and copied to clipboard!")

# Select all items
def select_all():
    tree.selection_set(tree.get_children())

# GUI Setup
root = tk.Tk()
root.title("🎨 Sentiment Analyzer Pro")
root.geometry("1500x800")
root.configure(bg='#1a1a1a')

# Make the root window responsive
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Main frame
main_frame = tk.Frame(root, bg='#1a1a1a', padx=20, pady=20)
main_frame.grid(row=0, column=0, sticky="nsew")

# Configure main frame grid weights
main_frame.grid_rowconfigure(5, weight=1)
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)
main_frame.grid_columnconfigure(2, weight=1)

# Title and subtitle
title_label = tk.Label(main_frame, text="Sentiment Analysis Pro", font=('Arial', 36, 'bold'), bg='#1a1a1a', fg='#ffffff')
title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20), sticky="ew")
subtitle_label = tk.Label(main_frame, text="Advanced Review Analysis with AI", font=('Arial', 18, 'italic'), bg='#1a1a1a', fg='#7f8c8d')
subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20), sticky="ew")

# Upload button
upload_btn = tk.Button(main_frame, text="📤 Upload Excel", command=load_and_process, bg='#2ecc71', fg='white', 
                      font=('Arial', 16, 'bold'), padx=20, pady=10, relief='raised', bd=3)
upload_btn.grid(row=2, column=1, pady=10, sticky="ew")

# Manual input section
input_frame = tk.Frame(main_frame, bg='#1a1a1a')
input_frame.grid(row=3, column=0, columnspan=3, pady=20, sticky="ew")
input_frame.grid_columnconfigure(1, weight=1)
entry_label = tk.Label(input_frame, text="📝 Enter Review:", font=('Arial', 24), bg='#1a1a1a', fg='#ffffff')
entry_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
entry = tk.Entry(input_frame, width=80, font=('Arial', 20), bg='#2c2c2c', fg='#ffffff', bd=3, insertbackground='#ffffff')
entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
predict_btn = tk.Button(input_frame, text="Predict", command=predict_manual, bg='#3498db', fg='white', 
                       font=('Arial', 16, 'bold'), padx=20, pady=10, relief='raised', bd=3)
predict_btn.grid(row=0, column=2, padx=10, pady=5, sticky="e")

# Result label
result_label = tk.Label(main_frame, text="", font=('Arial', 30, 'bold'), bg='#1a1a1a', fg='#2ecc71', pady=30)
result_label.grid(row=4, column=0, columnspan=3, pady=20, sticky="ew")

# Data display area
data_display_frame = tk.Frame(main_frame, bg='#1a1a1a')
data_display_frame.grid(row=5, column=0, columnspan=3, pady=20, sticky="nsew")
data_display_frame.grid_rowconfigure(0, weight=1)
data_display_frame.grid_columnconfigure(0, weight=2)
data_display_frame.grid_columnconfigure(1, weight=1)

# Treeview for Excel data
tree_frame = tk.Frame(data_display_frame, bg='#1a1a1a')
tree_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
tree_frame.grid_rowconfigure(0, weight=1)
tree_frame.grid_columnconfigure(0, weight=1)

# Add context menu
context_menu = tk.Menu(root, tearoff=0)
context_menu.add_command(label="Copy", command=copy_text)
context_menu.add_command(label="Cut", command=cut_text)
context_menu.add_separator()
context_menu.add_command(label="Select All", command=select_all)

def show_context_menu(event):
    context_menu.post(event.x_root, event.y_root)

columns = ("Review", "Actual", "Predicted", "Confidence")
tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=15)
tree.bind("<Button-3>", show_context_menu)

# Configure style
style = ttk.Style()
style.theme_use('clam')
style.configure("Custom.Treeview", 
               background="#2c2c2c", 
               foreground="#ffffff", 
               fieldbackground="#2c2c2c",
               font=('Arial', 12))
style.configure("Custom.Treeview.Heading", 
               background="#3498db", 
               foreground="white",
               font=('Arial', 14, 'bold'))
style.map("Custom.Treeview", 
          background=[('selected', '#3498db')],
          foreground=[('selected', 'white')])

for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=300, anchor='center', stretch=True)
tree.grid(row=0, column=0, sticky="nsew")

# Scrollbar for Treeview
scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
scrollbar.grid(row=0, column=1, sticky="ns")
tree.configure(yscrollcommand=scrollbar.set)

# Chart frame
chart_frame = tk.Frame(data_display_frame, bg='#2c2c2c', bd=2, relief=tk.SUNKEN)
chart_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
chart_frame.grid_rowconfigure(0, weight=1)
chart_frame.grid_columnconfigure(0, weight=1)

# Status label
status_label = tk.Label(main_frame, text="", bg='#1a1a1a', font=('Arial', 14, 'italic'), fg='#e74c3c')
status_label.grid(row=6, column=0, columnspan=3, pady=10, sticky="ew")

# Footer
footer = tk.Label(main_frame, text="Powered by xAI <3 | © 2023 Sentiment Analysis Pro", bg='#1a1a1a', 
                 fg='#7f8c8d', font=('Arial', 16, 'italic'))
footer.grid(row=7, column=0, columnspan=3, pady=20, sticky="ew")

# Load model and vectorizer on startup
try:
    model = load_model('compact_sentiment_model.h5')
    vectorizer = tf.keras.models.load_model('vectorizer')
    vectorizer = vectorizer.layers[0]
    status_label.config(text="✅ Model loaded successfully! Ready to analyze...", fg='#2ecc71')
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model or vectorizer: {str(e)}. Please ensure 'compact_sentiment_model.h5' and 'vectorizer' are in the folder.")
    root.quit()

root.mainloop()