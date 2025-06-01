import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import PyPDF2
from docx import Document
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
import threading

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define text cleaning function
def cleanResume(txt):
    cleanText = unicodedata.normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')
    cleanText = re.sub(r'http[s]?://\S+', ' ', cleanText)
    cleanText = re.sub(r'#[^\s]+', ' ', cleanText)
    cleanText = re.sub(r'@[^\s]+', ' ', cleanText)
    cleanText = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]', ' ', cleanText)
    cleanText = cleanText.lower()
    categories = [
        'java developer', 'testing', 'devops engineer', 'python developer', 'web designing',
        'hr', 'hadoop', 'blockchain', 'etl developer', 'operations manager', 'data science',
        'sales', 'mechanical engineer', 'arts', 'database', 'electrical engineering',
        'health and fitness', 'pmo', 'business analyst', 'dotnet developer', 'automation testing',
        'network security engineer', 'sap developer', 'civil engineer', 'advocate',
        'java', 'devops', 'python', 'etl', 'data scientist', 'business analysis', 'automation'
    ]
    for category in categories:
        cleanText = re.sub(rf'\b{category}\b', ' ', cleanText, flags=re.IGNORECASE)
    words = cleanText.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    cleanText = ' '.join(words)
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()
    return cleanText

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        messagebox.showerror("Error", f"Error reading PDF: {e}")
        return ""

# Function to extract text from Word document
def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        messagebox.showerror("Error", f"Error reading Word document: {e}")
        return ""

# Load model, vectorizer, and label encoder
def load_model():
    try:
        model = joblib.load('best_resume_classifier.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        le = joblib.load('label_encoder.pkl')
        print("Model, vectorizer, and label encoder loaded successfully!")
        return model, vectorizer, le
    except Exception as e:
        print(f"Error loading model: {e}")
        messagebox.showerror("Error", f"Error loading model: {e}")
        return None, None, None

model, vectorizer, le = load_model()

# Tkinter app
class ResumeClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resume Classifier for HR")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f4f8")

        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_frame.configure(style='TFrame')

        # Configure style
        style = ttk.Style()
        style.configure('TFrame', background="#f0f4f8")
        style.configure('TLabel', background="#f0f4f8", font=("Arial", 12))
        style.configure('TButton', font=("Arial", 10))
        style.configure('Header.TLabel', font=("Arial", 24, "bold"), foreground="#2c3e50")

        # Header
        self.header_label = ttk.Label(
            self.main_frame, text="📄 Resume Classifier for HR", style='Header.TLabel'
        )
        self.header_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Input frame
        self.input_frame = ttk.LabelFrame(self.main_frame, text="Resume Input", padding="10")
        self.input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        # Text area for resume input
        self.resume_text = tk.Text(self.input_frame, height=12, width=80, font=("Arial", 12), bg="#ffffff", fg="#2c3e50")
        self.resume_text.grid(row=0, column=0, columnspan=3, padx=5, pady=5)

        # Buttons
        self.upload_btn = ttk.Button(
            self.input_frame, text="📤 Upload File", command=self.upload_file
        )
        self.upload_btn.grid(row=1, column=0, padx=5, pady=5)
        self.upload_btn.bind('<Enter>', lambda e: self.show_tooltip(e, "Upload a PDF or Word resume"))
        self.upload_btn.bind('<Leave>', lambda e: self.hide_tooltip())

        self.clear_btn = ttk.Button(
            self.input_frame, text="🗑 Clear Text", command=self.clear_text
        )
        self.clear_btn.grid(row=1, column=1, padx=5, pady=5)
        self.clear_btn.bind('<Enter>', lambda e: self.show_tooltip(e, "Clear the resume text"))
        self.clear_btn.bind('<Leave>', lambda e: self.hide_tooltip())

        self.classify_btn = ttk.Button(
            self.input_frame, text="🔍 Classify Resume", command=self.start_classification
        )
        self.classify_btn.grid(row=1, column=2, padx=5, pady=5)
        self.classify_btn.bind('<Enter>', lambda e: self.show_tooltip(e, "Classify the resume"))
        self.classify_btn.bind('<Leave>', lambda e: self.hide_tooltip())

        # Progress bar
        self.progress = ttk.Progressbar(self.input_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.progress.grid_remove()

        # Results frame
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Classification Results", padding="10")
        self.results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        # Result label
        self.result_label = ttk.Label(
            self.results_frame, text="Predicted Category: Not classified yet", font=("Arial", 14), foreground="#2c3e50"
        )
        self.result_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        # Probabilities table
        self.tree = ttk.Treeview(self.results_frame, columns=("Category", "Probability"), show="headings", height=8)
        self.tree.heading("Category", text="Category")
        self.tree.heading("Probability", text="Probability (%)")
        self.tree.column("Category", width=300)
        self.tree.column("Probability", width=100)
        self.tree.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        # Export button
        self.export_btn = ttk.Button(
            self.results_frame, text="💾 Export to PDF", command=self.export_to_pdf
        )
        self.export_btn.grid(row=2, column=0, columnspan=2, pady=5)
        self.export_btn.bind('<Enter>', lambda e: self.show_tooltip(e, "Export results to PDF"))
        self.export_btn.bind('<Leave>', lambda e: self.hide_tooltip())

        # Plot frame
        self.plot_frame = ttk.LabelFrame(self.main_frame, text="Visualizations", padding="10")
        self.plot_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        # Plots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0)
        plt.tight_layout()

        # Tooltip label
        self.tooltip = None

    def show_tooltip(self, event, text):
        if self.tooltip:
            self.tooltip.destroy()
        x, y = self.root.winfo_pointerxy()
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x+10}+{y+10}")
        label = tk.Label(self.tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1)
        label.pack()

    def hide_tooltip(self):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf"), ("Word files", "*.docx")])
        if file_path:
            if file_path.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif file_path.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            if text:
                self.resume_text.delete(1.0, tk.END)
                self.resume_text.insert(tk.END, text)

    def clear_text(self):
        self.resume_text.delete(1.0, tk.END)
        self.result_label.config(text="Predicted Category: Not classified yet")
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.ax1.clear()
        self.ax2.clear()
        self.canvas.draw()

    def start_classification(self):
        if not model or not vectorizer or not le:
            messagebox.showerror("Error", "Model not loaded. Please ensure model files are available.")
            return

        resume_text = self.resume_text.get(1.0, tk.END).strip()
        if not resume_text:
            messagebox.showwarning("Warning", "Please provide a resume to classify.")
            return

        # Show progress bar
        self.progress.grid()
        self.progress.start()
        self.classify_btn.config(state='disabled')

        # Run classification in a separate thread
        threading.Thread(target=self.classify_resume, daemon=True).start()

    def classify_resume(self):
        resume_text = self.resume_text.get(1.0, tk.END).strip()
        cleaned_resume = cleanResume(resume_text)

        # Prepare input for model
        vector = vectorizer.transform([cleaned_resume])
        predicted_category = model.predict(vector)
        probabilities = model.predict_proba(vector)[0] if hasattr(model, 'predict_proba') else [1.0 if i == predicted_category else 0.0 for i in range(len(le.classes_))]

        # Decode prediction
        predicted_label = le.inverse_transform(predicted_category)[0]

        # Update UI in main thread
        self.root.after(0, lambda: self.update_results(predicted_label, probabilities, cleaned_resume))

    def update_results(self, predicted_label, probabilities, cleaned_resume):
        # Update result label
        self.result_label.config(text=f"Predicted Category: {predicted_label}")

        # Update probabilities table
        for item in self.tree.get_children():
            self.tree.delete(item)
        prob_df = pd.DataFrame({"Category": le.classes_, "Probability": [f"{p*100:.2f}%" for p in probabilities]})
        for _, row in prob_df.iterrows():
            self.tree.insert("", tk.END, values=(row["Category"], row["Probability"]))

        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Probability bar plot
        sns.barplot(x=probabilities, y=le.classes_, palette='Blues_d', ax=self.ax1)
        self.ax1.set_xlabel("Probability")
        self.ax1.set_ylabel("Category")
        self.ax1.set_title("Prediction Probabilities")

        # Word cloud
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate(cleaned_resume)
        self.ax2.imshow(wordcloud, interpolation='bilinear')
        self.ax2.axis('off')
        self.ax2.set_title("Key Words in Resume")

        # Redraw canvas
        self.canvas.draw()

        # Hide progress bar
        self.progress.stop()
        self.progress.grid_remove()
        self.classify_btn.config(state='normal')

        # Show success message
        messagebox.showinfo("Success", f"Resume classified as: {predicted_label}")

    def export_to_pdf(self):
        if not self.tree.get_children():
            messagebox.showwarning("Warning", "No results to export.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if not file_path:
            return

        # Save plots
        self.fig.savefig("temp_plots.png", bbox_inches='tight')

        # Create PDF
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        # Title
        elements.append(Paragraph("Resume Classification Report", styles['Title']))

        # Predicted category
        predicted_text = self.result_label.cget("text")
        elements.append(Paragraph(predicted_text, styles['Heading2']))
        elements.append(Paragraph("<br/>", styles['Normal']))

        # Probabilities table
        data = [["Category", "Probability"]]
        for item in self.tree.get_children():
            values = self.tree.item(item, 'values')
            data.append(list(values))
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#d3d3d3'),
            ('TEXTCOLOR', (0, 0), (-1, 0), '#000000'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), '#f5f5f5'),
            ('GRID', (0, 0), (-1, -1), 1, '#000000'),
        ]))
        elements.append(table)
        elements.append(Paragraph("<br/>", styles['Normal']))

        # Plots
        elements.append(Image("temp_plots.png", width=6*inch, height=4*inch))

        # Build PDF
        doc.build(elements)
        os.remove("temp_plots.png")
        messagebox.showinfo("Success", "Results exported to PDF!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ResumeClassifierApp(root)
    root.mainloop()