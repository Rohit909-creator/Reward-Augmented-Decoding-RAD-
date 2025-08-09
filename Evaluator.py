# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B-Instruct")


class Acknowledger():
    
    def __init__(self, LLM, RewardModel, max_steps):
        
        self.llm = LLM
        self.rm = RewardModel
        self.max_steps = max_steps
        
    def __call__(self):
        pass
    
    
    def generate(self):
        pass
    
    
