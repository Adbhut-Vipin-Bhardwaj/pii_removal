import os
import time
from huggingface_hub import login, HfApi
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM


token = "<huggingface token>"
login(token)

dataset_name = "ai4privacy/open-pii-masking-500k-ai4privacy"
model_name = "openai-community/gpt2"
chat_template = """{% for msg in messages %}
<{{ msg.role }}>{{ msg.content }}</{{ msg.role }}>
{% endfor %}
{% if add_generation_prompt %}
<pii_removed>{% endif %}"""
special_tokens = {"additional_special_tokens": ["<input>", "</input>", "<pii_removed>", "</pii_removed>"]}
output_dir = "./full_fine_tuning"
model_chkpts_dir = os.path.join(output_dir, "model_chkpts")
hf_repo_name = "Adbhut/gpt2_ft-ai4privacy-open-pii-masking-500k-ai4privacy"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.add_special_tokens(special_tokens)
# Set </pii_removed> as the EOS token
tokenizer.eos_token = "</pii_removed>"
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</pii_removed>")

# Make sure the pad_token is also set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.chat_template = chat_template

model.resize_token_embeddings(len(tokenizer))

def filter_by_lang(example):
    return example["language"] == "en"

def apply_template(example):
    input_text = example["source_text"]
    pii_removed = example["masked_text"]
    messages = [
        {"role": "input", "content": input_text},
        {"role": "pii_removed", "content": pii_removed},
    ]
    return {"messages": messages}

def save_to_hub(model, tokenizer, repo_name):
    api = HfApi(
        endpoint="https://huggingface.co",
        token=token,
    )
    api.create_repo(repo_id=repo_name, exist_ok=True)
    tokenizer.push_to_hub(repo_name)
    model.push_to_hub(repo_name)

total_ds = load_dataset(dataset_name)
train_val_ds = total_ds["train"].filter(filter_by_lang)
train_val_ds = train_val_ds.train_test_split(test_size=30000)
train_ds = train_val_ds["train"]
val_ds = train_val_ds["test"]
test_ds = total_ds["validation"].filter(filter_by_lang)

orig_col_names = train_ds.column_names
train_ds = train_ds.map(apply_template, remove_columns=orig_col_names)
val_ds = val_ds.map(apply_template, remove_columns=orig_col_names)

def generate_completion(messages, max_new_tokens=128):
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    tokens = tokenizer(prompt_text, return_tensors="pt")
    output = model.generate(
        **tokens, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id
    )
    completion = tokenizer.decode(output[0], skip_special_tokens=False)
    return completion

training_args = SFTConfig(
    output_dir=model_chkpts_dir,
    max_steps=22500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-3,
    eval_steps=500,
    eval_strategy="steps",
    save_steps=500,
    save_strategy="steps",
    save_only_model=True,
    save_total_limit=1,
    load_best_model_at_end=True,
    disable_tqdm=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
)

start = time.time()

trainer.train()

end = time.time()

time_taken = (end - start) / 60  # minutes
print(f"Time taken by training: {time_taken:.2f} minutes")

save_to_hub(model, tokenizer, hf_repo_name)
