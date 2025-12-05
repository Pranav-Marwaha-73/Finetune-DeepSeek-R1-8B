<h1>Finetuning DeepSeek-R1-8B for Medical Reasoning (LoRA + CoT)</h1>

<p>
  This project focuses on fine-tuning <strong>DeepSeek-R1-8B</strong> on a medical
  reasoning dataset <code>medical-o1-reasoning-SFT</code> using the
  <strong>LoRA</strong> technique. The model is trained to generate 
  <strong>Chain-of-Thought</strong> (CoT) reasoning before providing its final answer.
</p>

<h2>🔬 Project Overview</h2>
<ul>
  <li><strong>Base Model:</strong> DeepSeek-R1-8B</li>
  <li><strong>Dataset:</strong> medical-o1-reasoning-SFT</li>
  <li><strong>Fine-tuning Method:</strong> LoRA</li>
  <li><strong>Reasoning:</strong> Outputs chain-of-thought before final answer</li>
  <li><strong>Optimization:</strong> Unsloth for efficient fine-tuning</li>
  <li><strong>Experiment Tracking:</strong> Weights & Biases (W&B)</li>
  <li><strong>GPU Used:</strong> Tesla T4</li>
</ul>

<hr>

<h2>📊 Metrics</h2>
<ul>
  <li><strong>F1 Score:</strong> 0.85</li>
  <li><strong>Recall:</strong> 0.86</li>
  <li><strong>Precision:</strong> 0.84</li>
  <li><strong>Training Loss:</strong> 1.380500</li>
    <img width="1910" height="791" alt="Screenshot 2025-12-05 194927" src="https://github.com/user-attachments/assets/1ce6a158-019b-4852-8732-849c15970333" />
  <img width="1757" height="606" alt="Screenshot 2025-12-05 184642" src="https://github.com/user-attachments/assets/d737ca13-60be-4871-9c6d-bc7ee7578a3d" />

</ul>

<hr>

<h2>⚙️ Training Setup</h2>

<h3>LoRA Configuration</h3>
<pre>
lora_r: 8
lora_alpha: 16
lora_dropout: 0.1
modules: ["q_proj", "v_proj", "o_proj", "fc1", "fc2"]
</pre>

<h3>Training Hyperparameters</h3>
<pre>
batch_size: 8
micro_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2e-5
max_seq_length: 1024
optimizer: unsloth
epochs: 3
</pre>

<hr>

<h2>🧠 Chain-of-Thought Reasoning</h2>
<p>The model learns to output its reasoning process before giving the final answer.</p>

<pre>
Question: What is the first-line management of acute asthma?
Chain-of-thought: <step-by-step reasoning>
Answer: <final concise answer>
</pre>

<p>
Chain-of-thought enhances interpretability, especially for medical use cases.
</p>

<hr>

<h2>🚀 Inference Example (Python)</h2>

<pre>
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-r1-8b")
model = AutoModelForCausalLM.from_pretrained("YOUR_MODEL_PATH")

prompt = """
Question: A patient presents with severe shortness of breath. What is the immediate management?
Chain-of-thought:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=256)

print(tokenizer.decode(output[0], skip_special_tokens=True))
</pre>
<img width="1918" height="908" alt="Screenshot 2025-12-05 195718" src="https://github.com/user-attachments/assets/1e61e6c7-2cd1-4472-9cb0-51a1dd8e5033" />


<hr>

<h2>📈 Weights & Biases Logging</h2>

<pre>
import wandb
wandb.init(project="deepseek-medical-lora")

wandb.log({
  "train_loss": loss,
  "f1": f1,
  "recall": recall,
  "precision": precision
})
</pre>
<img width="1830" height="845" alt="Screenshot 2025-12-05 181520" src="https://github.com/user-attachments/assets/d12ee03f-3281-4ebb-8714-22224182dda9" />

<hr>

<h2>📌 Key Features</h2>
<ul>
  <li>Medical reasoning tuned through supervised CoT examples</li>
  <li>Lightweight LoRA adapters for efficient fine-tuning</li>
  <li>Optimized with Unsloth for speed and low VRAM use</li>
  <li>End-to-end experiment tracking with W&B</li>
</ul>
<img width="702" height="915" alt="Screenshot 2025-12-05 181010" src="https://github.com/user-attachments/assets/0eb4d39c-7795-4ad8-9c9c-5898c47169a2" />

<hr>

<h2>⚠️ Safety Notice</h2>
<p>
This model is not a certified medical system. All outputs must be reviewed by licensed medical professionals.
</p>

<hr>

<h2>📬 Contact</h2>
<p>
<strong>Author:</strong> Pranav Marwaha<br>
Feel free to use, modify, or contribute!
</p>
