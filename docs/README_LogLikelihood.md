# 🧠 Understanding Log-Likelihood in Language Models

Language models (like **Gemma**, **GPT**, or others) are trained to predict
“**what token comes next given the previous ones**.”
When we evaluate them, we often measure how *probable* a correct answer is under the model — this is called **log-likelihood**.

---

## ⚙️ Mathematical Definition

For a given input *x* and correct answer *y = (y₁, y₂, ..., yₜ)*(a sequence of tokens),
the probability of the answer is:

\**P(y|x) = ∏ₜ₌₁ᵀ P(yₜ | x, y₍₍ₜ₋₁₎₎)**

Taking the logarithm gives the **log-likelihood**:

\**log P(y|x) = Σₜ₌₁ᵀ log P(yₜ | x, y₍₍ₜ₋₁₎₎)**

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

> چه همکاری‌ای در دههٔ ۱۳۶۰ به تحول ایران‌خودرو کمک کرد؟

and the correct answer is:

> اعلام آمادگی پژو برای همکاری

We break the answer into tokens and get model probabilities like this:


<div dir="rtl">

| 🔢 گام | 🪶 توکن | 🔢 احتمال \(P(y_t | x, y_{<t})\) | 🧮 لگاریتم احتمال \(\log P(y_t | x, y_{<t})\) |
|:--:|:---------------------------|:----------------:|:----------------:|
| ۱ | اعلام (*announcement*) | 0.25 | −1.386 |
| ۲ | آمادگی (*readiness*) | 0.40 | −0.916 |
| ۳ | پژو (*Peugeot*) | 0.10 | −2.302 |
| ۴ | برای (*for*) | 0.30 | −1.203 |
| ۵ | همکاری (*cooperation*) | 0.50 | −0.693 |

</div>


Then:

**P(y | x) = 0.25 × 0.40 × 0.10 × 0.30 × 0.50 = 0.0015**

**log P(y | x) = −1.386 − 0.916 − 2.302 − 1.203 − 0.693 = −6.5**

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
