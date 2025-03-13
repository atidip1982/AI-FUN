import tkinter as tk
from tkinter import ttk, messagebox
from sentence_transformers import SentenceTransformer, util

# Load SBERT Model
model = SentenceTransformer('all-MiniLM-L6-v2')


# Function to compare texts and calculate similarity
def compare_texts():
    question = question_entry.get().strip()
    chat_gpt_text = chat_gpt_text_area.get("1.0", tk.END).strip()
    inhouse_gpt_text = inhouse_gpt_text_area.get("1.0", tk.END).strip()

    if not question or not chat_gpt_text or not inhouse_gpt_text:
        messagebox.showwarning("Input Error", "Please fill all fields!")
        return

    # Compute similarity score
    embedding1 = model.encode(chat_gpt_text, convert_to_tensor=True)
    embedding2 = model.encode(inhouse_gpt_text, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()

    # Convert to percentage
    relevancy_percentage = round(similarity_score * 100, 2)

    # Insert data into table
    result_table.insert("", "end", values=(question, chat_gpt_text, inhouse_gpt_text, f"{relevancy_percentage}%"))


# Create the main window
root = tk.Tk()
root.title("GPT Response Comparison Tool")
root.geometry("850x550")
root.configure(bg="#f0f0f0")

# === Main Frame with Borders ===
main_frame = tk.Frame(root, bg="white", bd=2, relief="ridge")
main_frame.pack(padx=20, pady=20, fill="both", expand=True)

# === Input Panel ===
input_frame = tk.Frame(main_frame, bg="white", padx=10, pady=10)
input_frame.pack(fill="x")

tk.Label(input_frame, text="Enter Question:", font=("Arial", 10, "bold"), bg="white").pack(anchor="w")
question_entry = tk.Entry(input_frame, font=("Arial", 12), bd=2, relief="solid", width=80)
question_entry.pack(pady=5)

tk.Label(input_frame, text="ChatGPT Response:", font=("Arial", 10, "bold"), bg="white").pack(anchor="w")
chat_gpt_text_area = tk.Text(input_frame, height=4, width=80, bd=2, relief="solid", wrap="word")
chat_gpt_text_area.pack(pady=5)

tk.Label(input_frame, text="In-house GPT Response:", font=("Arial", 10, "bold"), bg="white").pack(anchor="w")
inhouse_gpt_text_area = tk.Text(input_frame, height=4, width=80, bd=2, relief="solid", wrap="word")
inhouse_gpt_text_area.pack(pady=5)

# === Compare Button ===
compare_button = ttk.Button(main_frame, text="Compare & Add", command=compare_texts)
compare_button.pack(pady=10)

# === Table for Results ===
table_frame = tk.Frame(main_frame, bg="white", bd=2, relief="solid")
table_frame.pack(fill="both", expand=True, padx=10, pady=10)

columns = ("Question", "ChatGPT Response", "In-house GPT Response", "Relevancy Score (%)")
result_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)

# Set column headings
for col in columns:
    result_table.heading(col, text=col)
    result_table.column(col, width=200, anchor="center")

# Scrollbars for table
scroll_y = ttk.Scrollbar(table_frame, orient="vertical", command=result_table.yview)
scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=result_table.xview)
result_table.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

scroll_y.pack(side="right", fill="y")
scroll_x.pack(side="bottom", fill="x")
result_table.pack(fill="both", expand=True)

# Run the Tkinter event loop
root.mainloop()
