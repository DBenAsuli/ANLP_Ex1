# Advanced Natural Language Processing     Exercise 1
# Dvir Ben Asuli                           318208816
# The Hebrew University of Jerusalem       June 2024

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

dataset = load_dataset("nyu-mll/glue", "ax")

# Full Fine-tune
def q1():
    model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
    pass

# LoRA Fine-tune
def q2():
    model = AutoModel.from_pretrained("microsoft/deberta-v3-base")

# Bigger models
def q3():
    model1 = AutoModel.from_pretrained("microsoft/deberta-v3-large")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    model2 = AutoModelForCausalLM.from_pretrained("google/gemma-2b")


if __name__ == '__main__':
    question = input("Please enter question number: ")
    # question = "1" #FIXME REMOVE

    if question == "1":
        print("Starting Question 1- Full Fine-Tune")
        pass
    elif question == "2":
        print("Starting Question 2- LoRA Fine-Tune")
        pass
    elif question == "3":
        print("Starting Question 2- Bigger Models")
        pass
    else:
        print("Invalid Question Number (should be 1, 2 or 3)")
