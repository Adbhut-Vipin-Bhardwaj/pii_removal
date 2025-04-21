import argparse
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer


token = "<huggingface token>"
login(token)


model_name = "Adbhut/gpt2_ft-ai4privacy-open-pii-masking-500k-ai4privacy"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def generate_completion(messages, max_new_tokens=128):
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    tokens = tokenizer(prompt_text, return_tensors="pt")
    output = model.generate(
        **tokens, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    completion = tokenizer.decode(output[0], skip_special_tokens=False)
    return completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mnt", "--max_new_tokens", type=int, default=50,
        help="Maximum number of new tokens to generate (default: 50)"
    )
    args = parser.parse_args()

    while True:
        prompt = input("Prompt (Enter q to exit): ")

        if prompt == "q":
            break

        completion = generate_completion(
            [
                {"role": "input", "content": prompt},
            ],
            max_new_tokens=args.max_new_tokens
        )
        print(f"Completion: {completion}")


if __name__ == "__main__":
    main()
