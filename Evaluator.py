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
        
    def get_reward_adjusted_logits(self, original_logits, question, current_context, top_k_indices, alpha=2.0):
        """RAD implementation: adjust logits based on reward signals"""
        reward_scores = []
        
        for token_id in top_k_indices:
            # Decode candidate token
            candidate_token = self.tokenizer.decode([token_id])
            candidate_context = current_context + candidate_token
            
            # Get reward score for this candidate
            inputs = rm_tokenizer(question, candidate_context, return_tensors="pt", truncation=True, max_length = 512)
            with torch.no_grad():
                reward_logits = self.rm(**inputs).logits.squeeze()
                reward_score = reward_logits.item()
            reward_scores.append(reward_score)
        
        # Convert to tensor and adjust original logits
        reward_tensor = torch.tensor(reward_scores, device=original_logits.device)
        adjusted_logits = original_logits + alpha * reward_tensor
        
        return adjusted_logits, reward_scores
        
    def __call__(self, text):
        pass
        
    def generate(self, text, max_new_tokens=10, temperature=1.0, top_k=50, top_p=0.5, alpha=2.0, enable_rad=True):
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
        rad_rewards = []  # Track RAD reward signals
        
        print(f"🤖 Generating response with {'RAD' if enable_rad else 'Standard'} decoding...")
        print("📝 Generated text: ", end="", flush=True)
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]  # Take last token's logits

                # Apply temperature scaling
                if temperature != 1.0:
                    logits = logits / temperature

                # Top-k filtering - get candidates first
                if top_k is not None:
                    top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    
                    # RAD MODIFICATION: Adjust logits with reward signals
                    if enable_rad and self.rm is not None:
                        adjusted_logits, token_rewards = self.get_reward_adjusted_logits(
                            top_k_values, text, context, top_k_indices.squeeze().tolist(), alpha
                        )
                        rad_rewards.extend(token_rewards)  # Store for analysis
                        
                        # Use adjusted logits
                        top_k_values = adjusted_logits
                    
                    # Apply top-k mask
                    mask = torch.full_like(logits, float("-inf"))
                    logits = mask.scatter(1, top_k_indices, top_k_values)

                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Create mask for tokens to keep (cumulative prob <= top_p)
                    sorted_mask = cumulative_probs <= top_p
                    
                    # Ensure at least the first token is kept
                    sorted_mask[..., 0] = True
                    
                    # Create inverse mask for scattering
                    indices_to_remove = ~sorted_mask
                    
                    # Scatter back to original positions
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    mask = mask.scatter(1, sorted_indices, indices_to_remove)
                    logits = logits.masked_fill(mask, float("-inf"))

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append next token to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

                # STREAMING OUTPUT: Print token immediately
                new_token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                print(new_token_text, end="", flush=True)

                # Stop if EOS is generated
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
                # Update context and get reward score
                context = tokenizer.decode(input_ids[0, inputs["input_ids"].shape[1]:])
                
                # Get overall reward score for progress tracking
                reward_inputs = rm_tokenizer(text, context, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    reward_logits = rm(**reward_inputs).logits.squeeze()
                    score = reward_logits.item()
                scores.append(score)
                steps.append(i)

        print()  # New line after generation
        
        # Decode only the newly generated part
        generated_ids = input_ids[0, inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "generated_ids": context,
            "generated_text": generated_text,
            "steps": steps,
            "scores": scores,
            "rad_rewards": rad_rewards,  # NEW: RAD reward signals
            "method": "RAD" if enable_rad else "Standard"
        }
        

def detect_and_correct_hallucination(self, question, full_context, current_step, max_tokens, correction_threshold=0.3):
    """
    Detect hallucination by comparing expected vs actual trajectory
    Returns: (should_correct, correction_prompt, severity)
    """
    # Only check after we have enough context (your 50% rule)
    if current_step < max_tokens * 0.5:
        return False, "", 0.0
    
    # Split context for analysis (your 50/25/25 idea)
    tokens = self.tokenizer.encode(full_context)
    total_generated = len(tokens)
    
    if total_generated < 20:  # Need minimum context
        return False, "", 0.0
    
    # Get the "stable" prefix (first 50% of generation)
    prefix_length = int(total_generated * 0.5)
    recent_length = min(int(total_generated * 0.25), total_generated - prefix_length)
    
    prefix_tokens = tokens[:prefix_length]
    recent_tokens = tokens[prefix_length:prefix_length + recent_length]
    
    prefix_text = self.tokenizer.decode(prefix_tokens)
    recent_text = self.tokenizer.decode(recent_tokens)
    
    # Get expected trajectory reward (what SHOULD come next)
    expected_score = self.get_reward_score(question, prefix_text)
    
    # Get actual trajectory reward (what DID come next)  
    actual_score = self.get_reward_score(question, prefix_text + recent_text)
    
    # Calculate divergence (your core insight!)
    reward_divergence = expected_score - actual_score
    
    # Additional semantic divergence check
    # Check if recent generation is semantically consistent
    consistency_score = self.get_reward_score(question, recent_text)  # Standalone coherence
    
    # Combined divergence metric
    total_divergence = reward_divergence + max(0, expected_score - consistency_score) * 0.5
    
    # Determine if correction is needed
    should_correct = total_divergence > correction_threshold
    
    # Determine correction type based on severity
    if total_divergence > correction_threshold * 2:
        correction_prompt = " Oh sorry, let me correct that hallucination and provide accurate information: "
        severity = "hard"
    elif should_correct:
        correction_prompt = " Actually, let me clarify that better: "
        severity = "soft"
    else:
        correction_prompt = ""
        severity = "none"
    
    return should_correct, correction_prompt, total_divergence    

    
    
if __name__ == "__main__":
    
    ack = Acknowledger(RewardModel=rm)  # Pass the reward model
    
    question = "who is the president of mars?"
    print(f"🔬 Question: {question}")
    print("="*60)
    
    # Generate with RAD
    print("\n🚀 RAD Generation:")
    rad_result = ack.generate(
        question, 
        max_new_tokens=50, 
        enable_rad=True, 
        alpha=1.5,  # Moderate RAD influence
        top_k=20
    )
    
    # Generate without RAD for comparison
    print(f"\n🔄 Standard Generation:")
    std_result = ack.generate(
        question, 
        max_new_tokens=50, 
        enable_rad=False,
        top_k=20
    )
    
    with open("Generations.txt", "a") as f:
        s = f"RAD Generation: {rad_result['generated_text']}\n\nStandard Generation: {std_result['generated_text']}\nRAD_Final Score: {rad_result['scores'][-1]}\nStandard Final Score: {std_result['scores'][-1]}"
        f.write(s)
    
    # Analysis and visualization
    print(f"\n📊 Results:")
    print(f"RAD Final Score: {rad_result['scores'][-1]:.3f}")
    print(f"Standard Final Score: {std_result['scores'][-1]:.3f}")
    print(f"RAD Improvement: {rad_result['scores'][-1] - std_result['scores'][-1]:+.3f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rad_result["steps"], rad_result["scores"], 'b-o', label='RAD', linewidth=2)
    plt.plot(std_result["steps"], std_result["scores"], 'r--s', label='Standard', linewidth=2)
    plt.xlabel('Generation Step')
    plt.ylabel('Reward Score')
    plt.title('Reward Progression Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if rad_result['rad_rewards']:
        plt.hist(rad_result['rad_rewards'], bins=20, alpha=0.7, color='blue', label='RAD Token Rewards')
        plt.xlabel('Individual Token Reward')
        plt.ylabel('Frequency')
        plt.title('Distribution of RAD Token Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("rad_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()