# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.set_default_device("cuda")

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B-Instruct")

model_name = "OpenAssistant/reward-model-deberta-v3-base"
rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
rm = AutoModelForSequenceClassification.from_pretrained(model_name)  # ~738MB file

class Acknowledger():
    
    def __init__(self, LLM=model, RewardModel=None, max_steps=10):
        
        self.llm = model
        self.tokenizer = tokenizer
        self.rm = RewardModel
        self.max_steps = max_steps
        
    def __call__(self, text):
        pass
        
    def generate(self, text, max_new_tokens=10, temperature=1.0, top_k=None, top_p=0.5):
        # Prepare initial input using chat template
        messages = [{"role": "user", "content": text}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.llm.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        self.llm.eval()  # Ensure eval mode

        scores = []
        steps = []
        context = ""
        with torch.no_grad():
            for i in range(max_new_tokens):
                outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]  # Take last token's logits

                # Apply temperature scaling
                if temperature != 1.0:
                    logits = logits / temperature

                # Top-k filtering
                if top_k is not None:
                    values, indices = torch.topk(logits, top_k)
                    mask = torch.full_like(logits, float("-inf"))
                    logits = mask.scatter(1, indices, values)

                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_mask = cumulative_probs > top_p
                    sorted_mask_clone = sorted_mask.clone()
                    sorted_mask_clone[..., 1:] = sorted_mask_clone[..., :-1]
                    
                    sorted_mask_clone[..., 0] = 0

                    sorted_mask = sorted_mask_clone

                    mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                    logits = logits.masked_fill(mask, float("-inf"))

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append next token to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

                # Stop if EOS is generated
                if next_token.item() == tokenizer.eos_token_id:
                    break
                context = tokenizer.decode(input_ids[0, inputs["input_ids"].shape[1]:])
                inputs = rm_tokenizer(text, context, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    logits = rm(**inputs).logits.squeeze()   # scalar score (higher -> more strongly preferred)
                    score = logits.item()
                scores.append(score)
                steps.append(i)


        # Decode only the newly generated part
        generated_ids = input_ids[0, inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "generated_ids": context,
            "generated_text": generated_text,
            "steps": steps,
            "scores":scores
        }

    
    
if __name__ == "__main__":
    
    ack = Acknowledger()
    result = ack.generate("What is photosynthesis?", max_new_tokens=250)
    print(result["generated_ids"])
    print(result["generated_text"])
    
    plt.plot(result["steps"], result["scores"])
    plt.savefig("hallucination_graph.png")    
    
    