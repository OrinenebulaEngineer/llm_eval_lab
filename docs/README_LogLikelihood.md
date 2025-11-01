# 🧠 Understanding Log-Likelihood in Language Models

Language models (like **Gemma**, **GPT**, or others) are trained to predict
“**what token comes next given the previous ones**.”
When we evaluate them, we often measure how *probable* a correct answer is under the model — this is called **log-likelihood**.

---

## ⚙️ Mathematical Definition

For a given input \( x \) (the context or question)
and the correct answer \( y = (y_1, y_2, ..., y_T) \) (a sequence of tokens),
the probability of the answer is:

\[
P(y | x) = \prod_{t=1}^{T} P(y_t | x, y_{<t})
\]

Taking the logarithm gives the **log-likelihood**:

\[
\log P(y | x) = \sum_{t=1}^{T} \log P(y_t | x, y_{<t})
\]

This value tells us how *confident* the model is in the gold answer.

---

## 💡 Intuitive Explanation

The model never “sees” the full answer.During evaluation, the system feeds the model the **context plus the answer up to token \(t-1\)**and asks:

> “If you knew this much, how likely is the next token?”

The model doesn’t generate — it *evaluates* the likelihood of the correct continuation.

At each step:

- The prefix (context + previous tokens) is given to the model.
- The model predicts probabilities for *all* possible next tokens.
- We extract the probability of the **true next token**.

---

## 📊 Example Calculation

Suppose the question is:

> *What collaboration helped Iran Khodro transform in the 1980s?*

and the correct answer is:

> *Peugeot’s cooperation announcement.*

We break the answer into tokens and get model probabilities like this:

| t | Token | \(P(y_t | x, y_{<t})\) | \(\log P(y_t | x, y_{<t})\) |
|---|--------|------------------|---------------|
| 1 | اعلام (*announcement*) | 0.25 | −1.386 |
| 2 | آمادگی (*readiness*) | 0.40 | −0.916 |
| 3 | پژو (*Peugeot*) | 0.10 | −2.302 |
| 4 | برای (*for*) | 0.30 | −1.203 |
| 5 | همکاری (*cooperation*) | 0.50 | −0.693 |

Then:

\[
P(y|x) = 0.25 × 0.40 × 0.10 × 0.30 × 0.50 = 0.0015
\]

\[
\log P(y|x) = -1.386 - 0.916 - 2.302 - 1.203 - 0.693 = -6.5
\]

So the **log-likelihood = −6.5**.

- If it’s **close to zero**, the model is *confident* and likely correct.
- If it’s **very negative**, the model is uncertain or wrong.

---

## 🔍 Step-by-Step Process

| Step | Model Input                                           | Model Predicts               | We Take                 |
| ---- | ----------------------------------------------------- | ---------------------------- | ----------------------- |
| 1    | `context`                                           | probabilities for all tokens | log P(“اعلام”)   |
| 2    | `context + اعلام`                              | probabilities for all tokens | log P(“آمادگی”) |
| 3    | `context + اعلام آمادگی`                 | probabilities for all tokens | log P(“پژو”)       |
| 4    | `context + اعلام آمادگی پژو`          | probabilities for all tokens | log P(“برای”)     |
| 5    | `context + اعلام آمادگی پژو برای` | probabilities for all tokens | log P(“همکاری”) |

This continues until all tokens in the gold answer are covered.

---

## 💻 Implementation in PyTorch

Here’s a minimal working example using the Hugging Face `transformers` library:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

context = "In the 1980s, Iran Khodro faced a major economic crisis..."
answer = "Peugeot announced its readiness to cooperate"

input_text = context + " " + answer
tokens = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**tokens)
    logits = outputs.logits[:, :-1, :]  # Predictions for next tokens
    labels = tokens.input_ids[:, 1:]    # True next tokens

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_likelihoods = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    log_likelihood = token_log_likelihoods.sum().item()

print(f"Total log-likelihood: {log_likelihood:.2f}")
```

This implements the formula:

\[
\log P(y|x) = \sum_{t=1}^{T} \log P(y_t | x, y_{<t})
\]

---

## 📈 Relation to Evaluation in `lm-evaluation-harness`

In evaluation tools like [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness),log-likelihood is used for QA tasks such as **SQuAD** to calculate:

- **Exact Match (EM)**
- **F1 score**
- Confidence-based ranking of predictions

Each record logs something like:

```json
["-33.25", "False"]
```

Meaning:

- Log-likelihood = −33.25
- Model’s predicted answer did **not** match the gold answer.

---

## 🧮 Summary

| Concept        | Meaning                    | Interpretation                                   |
| -------------- | -------------------------- | ------------------------------------------------ |
| \(P(y          | x)\)                       | Probability the model assigns to the gold answer |
| Log-likelihood | Sum of log P of all tokens | Measures model confidence                        |
| Closer to 0    | Model is confident         | Likely correct                                   |
| Very negative  | Model uncertain            | Likely wrong                                     |

---

## 📚 References

- [EleutherAI / lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers/)
- [Google Gemma Models](https://huggingface.co/google/gemma-2b-it)
