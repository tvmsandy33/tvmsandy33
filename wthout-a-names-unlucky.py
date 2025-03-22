# %% [code] {"execution":{"iopub.status.busy":"2025-03-22T23:21:44.343870Z","iopub.execute_input":"2025-03-22T23:21:44.344220Z","iopub.status.idle":"2025-03-22T23:21:46.587112Z","shell.execute_reply.started":"2025-03-22T23:21:44.344168Z","shell.execute_reply":"2025-03-22T23:21:46.586094Z"},"jupyter":{"outputs_hidden":false}}
import pandas as pd
from datasets import Dataset
df = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv')
df = df[['problem', 'answer']]
df.rename(columns={'problem':'question'}, inplace=True)
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2025-03-22T23:21:46.588528Z","iopub.execute_input":"2025-03-22T23:21:46.588888Z","iopub.status.idle":"2025-03-22T23:21:46.604345Z","shell.execute_reply.started":"2025-03-22T23:21:46.588856Z","shell.execute_reply":"2025-03-22T23:21:46.603296Z"},"jupyter":{"outputs_hidden":false}}
df['answer'] = df['answer'].astype(str)
df['answer'] = '#### ' + df['answer']

# Ensure question is string as well
df['question'] = df['question'].astype(str)

# Convert to HF Dataset
custom_dataset = Dataset.from_pandas(df)

# %% [code] {"execution":{"iopub.status.busy":"2025-03-22T19:51:00.780531Z","iopub.execute_input":"2025-03-22T19:51:00.780742Z","iopub.status.idle":"2025-03-22T19:51:29.335914Z","shell.execute_reply.started":"2025-03-22T19:51:00.780725Z","shell.execute_reply":"2025-03-22T19:51:29.335214Z"},"jupyter":{"outputs_hidden":false}}
# 📝 LaTRO Kaggle Notebook with Full RLOO + SFT Loss + KL Penalty + Save Model + Full Truncation + Chat Template Logic

from dataclasses import dataclass, field
from typing import Optional, List, Literal
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from accelerate import Accelerator

# --- CONFIG ---
@dataclass
class TrainerConfig:
    exp_name: str = "latro_kaggle_exp_1"
    run_name: Optional[str] = None
    model_name_or_path: str = '/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/2'#"EleutherAI/pythia-160m"
    dataset_name: Literal["gsm8k"] = "gsm8k"
    total_episodes: Optional[int] = None
    num_train_epochs: int = 1
    num_evaluations: int = 2
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    rollout_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-7
    rloo_k: int = 2
    kl_coef: float = 0.05
    response_length: int = 200
    stop_token: Optional[str] = "both"
    stop_token_ids: Optional[List[int]] = None
    temperature: float = 1.0
    sft_penalty: float = 0.1
    sanity_check: bool = True

config = TrainerConfig()

# --- TOKENIZER + MODEL ---
tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, torch_dtype=torch.float16)
ref_model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, torch_dtype=torch.float16).to('cpu')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

stop_sequences = ["The answer is", "####", "Answer:"]
stop_token_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]

# --- CHAT TEMPLATE LOGIC ---
def apply_chat_template(question: str) -> str:
    messages = [{"role": "system", "content": "Think step-by-step to arrive at the correct answer. Write down each thinking step. Only keep a minimum draft for each thinking step, with 5 words at most. Return final answer within \\boxed{}, after taking modulo 1000."},
                {"role": "user", "content": question},
            ]
    return tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True)

# --- DATASET ---
def prepare_dataset_gsm8k(dataset: Dataset, tokenizer):
    queries = [apply_chat_template(q) for q in dataset["question"]]
    responses = [f"The answer is {a.split('#### ')[-1].strip()}" for a in dataset["answer"]]
    return Dataset.from_dict({"queries": queries, "responses": responses})

#raw = load_dataset("openai/gsm8k", name="main")
train_dataset = prepare_dataset_gsm8k(custom_dataset.select(range(8)), tokenizer)
eval_dataset = prepare_dataset_gsm8k(custom_dataset.select(range(8,10)), tokenizer)

# --- TRAINER ---
class FullLatroTrainer:
    def __init__(self, model, ref_model, tokenizer, train_ds, eval_ds, config):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.config = config
        self.accelerator = Accelerator()

        self.train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        self.eval_loader = DataLoader(eval_ds, batch_size=1)

        self.model = self.accelerator.prepare(self.model)
        self.ref_model = self.ref_model.to(self.accelerator.device)
        self.model.gradient_checkpointing_enable()


    def truncate_response(self, sequences):
        truncated = []
        for seq in sequences:
            found = False
            for stop_ids in stop_token_ids:
                stop_len = len(stop_ids)
                for i in range(len(seq) - stop_len + 1):
                    if torch.equal(seq[i:i+stop_len], torch.tensor(stop_ids, device=seq.device)):
                        truncated.append(torch.cat([seq[:i+stop_len], torch.full((len(seq)-i-stop_len,), tokenizer.pad_token_id, device=seq.device)]))
                        found = True
                        break
                if found: 
                    break
            if not found:
                truncated.append(seq)
        return torch.stack(truncated)

    def compute_logprobs(self, model, inputs, labels, context_lengths):
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1]

        # Align log_probs length with target length
        target = labels["input_ids"][:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs[:, -target.shape[1]:, :]

        mask = torch.arange(target.shape[1]).unsqueeze(0).to(target.device) >= context_lengths.unsqueeze(1)
        loss = F.nll_loss(log_probs.reshape(-1, log_probs.size(-1)), target.reshape(-1), reduction='none')
        loss = loss.view(target.shape)
        loss = loss.masked_fill(mask, 0.0)

        return loss.sum(1), log_probs, target, mask

    def train(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        for epoch in range(self.config.num_train_epochs):
            for batch in self.train_loader:
                inputs = self.tokenizer(batch["queries"], return_tensors="pt", padding=True).to(self.accelerator.device)
                labels = self.tokenizer(batch["responses"], return_tensors="pt", padding=True).to(self.accelerator.device)
                context_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)

                # Repeat queries and labels for RLOO sampling
                inputs["input_ids"] = inputs["input_ids"].repeat_interleave(config.rloo_k, dim=0)
                labels["input_ids"] = labels["input_ids"].repeat_interleave(config.rloo_k, dim=0)
                context_lengths = context_lengths.repeat_interleave(config.rloo_k, dim=0)

                rewards = []
                ref_rewards = []
                for _ in range(config.rloo_k):
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    generation = unwrapped_model.generate(**inputs, max_new_tokens=config.response_length)
                    generation = self.truncate_response(generation)

                    model_lp, model_log_probs, targets, mask = self.compute_logprobs(self.model, {"input_ids": generation}, labels, context_lengths)
                    with torch.no_grad():
                        ref_lp, _, _, _ = self.compute_logprobs(self.ref_model, {"input_ids": generation}, labels, context_lengths)

                    rewards.append(-model_lp)
                    ref_rewards.append(-ref_lp)

                rewards = torch.stack(rewards, dim=1)
                ref_rewards = torch.stack(ref_rewards, dim=1)

                baseline = (rewards.sum(1, keepdim=True) - rewards) / (config.rloo_k - 1)
                advantages = rewards - baseline

                # KL penalty
                kl_penalty = -config.kl_coef * (rewards - ref_rewards)
                advantages += kl_penalty

                rloo_loss = (advantages * rewards).mean()

                # SFT loss
                sft_loss, *_ = self.compute_logprobs(self.model, inputs, labels, context_lengths)
                sft_loss = sft_loss.mean()

                total_loss = rloo_loss + config.sft_penalty * sft_loss
                self.accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()
                print(f"Epoch {epoch} RLOO Loss: {rloo_loss.item():.4f} | SFT Loss: {sft_loss.item():.4f} | KL Penalty: {kl_penalty.mean().item():.4f}")

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.eval_loader:
                inputs = self.tokenizer(batch["queries"], return_tensors="pt", padding=True).to(self.accelerator.device)
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                generated = unwrapped_model.generate(**inputs, max_new_tokens=500)
                truncated = self.truncate_response(generated)
                decoded = self.tokenizer.batch_decode(truncated, skip_special_tokens=True)
                print(f"Q: {batch['queries'][0]} \nA: {decoded[0]}")

    def save_model(self, save_dir="./saved_model"):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            print(f"Model saved to {save_dir}")

# %% [code] {"execution":{"iopub.status.busy":"2025-03-22T19:51:29.336727Z","iopub.execute_input":"2025-03-22T19:51:29.336940Z","iopub.status.idle":"2025-03-22T19:51:31.384546Z","shell.execute_reply.started":"2025-03-22T19:51:29.336922Z","shell.execute_reply":"2025-03-22T19:51:31.383807Z"},"jupyter":{"outputs_hidden":false}}
# --- RUN TRAINING ---
trainer = FullLatroTrainer(model, ref_model, tokenizer, train_dataset, eval_dataset, config)

# %% [code] {"execution":{"iopub.status.busy":"2025-03-22T19:51:31.385262Z","iopub.execute_input":"2025-03-22T19:51:31.385483Z","iopub.status.idle":"2025-03-22T19:51:56.011899Z","shell.execute_reply.started":"2025-03-22T19:51:31.385465Z","shell.execute_reply":"2025-03-22T19:51:56.010870Z"},"jupyter":{"outputs_hidden":false}}
trainer.train()

# %% [code] {"execution":{"iopub.status.busy":"2025-03-22T20:29:50.707367Z","iopub.execute_input":"2025-03-22T20:29:50.707713Z"},"jupyter":{"outputs_hidden":false}}
trainer.evaluate()
trainer.save_model("./latro_final_checkpoint")