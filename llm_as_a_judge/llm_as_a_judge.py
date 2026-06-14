"""
llm_as_a_judge.py
=================
Standalone Python module for the LLM-as-a-Judge evaluation protocol used in:

  "From Vectors to Subspaces: Gaussian Concept Control for
   Customizable AI Assistant Personalities"
  Renata R. Latypova, Innopolis University, 2026.

Each generated text is scored on two independent dimensions (0-10) by a
judge LLM (paper: Qwen2.5-Max).  Every text is evaluated N_EVALS times
(default 3) and the scores are averaged to reduce variance.

Dimensions
----------
- Emotional intensity  : how strongly the target emotion is present
- Coherence            : grammatical and semantic quality of the text

Usage
-----
1. As a script (most common):

    python llm_as_a_judge.py \\
        --csv        generation_results.csv \\
        --emotion    SAD \\
        --api-url    https://your-openai-compatible-endpoint/v1/chat/completions \\
        --api-key    $YOUR_API_KEY \\
        --model      Qwen/Qwen2.5-Max \\
        --output-dir results/

2. As a library:

    from llm_as_a_judge import LLMJudge, JUDGE_PROMPTS

    judge = LLMJudge(api_url="...", api_key="...", model_name="...", emotion="SAD")
    emotional_score, coherence_score = judge.evaluate("Some generated text.")

Input CSV
---------
Must contain at least these columns (produced by gcs/, agcs/, kcs/ scripts):
  response          – generated text to evaluate
  sigma_level       – steering strength (renamed to `strength` internally)
  response_length   – number of tokens in the response

Output
------
results/
  llm_judge_results.csv   – row-level scores (all evaluations)
  llm_judge_summary.csv   – mean scores grouped by steering strength
  {EMOTION}_combined_scores.png
  {EMOTION}_steering_heatmap.png
"""

from __future__ import annotations

import argparse
import os
import re
import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Judge prompts  (one per target emotion + coherence; see Appendix B of thesis)
# ---------------------------------------------------------------------------

JUDGE_PROMPTS: dict[str, str] = {
    "SAD": """\
You are an expert evaluator assessing text for sadness intensity and quality.

Rate the following text on TWO dimensions (0-10 scale):

### 1. SAD_SCORE (Emotional Sadness)
- 0-2: Joyful, uplifting, optimistic content
- 3-4: Slightly joyful, funny
- 5: Neutral, no clear emotional valence
- 6-7: Slightly melancholic or bittersweet, but not truly sad
- 8-9: Noticeably sad, evokes sympathy or pity
- 10: Deeply sorrowful, heartbreaking, profound grief or despair

### 2. COHERENCE_SCORE (Text Quality)
- 0: Completely broken, gibberish, model collapsed
- 1-3: Severely broken, repetitive, nonsensical
- 4-5: Partially coherent but with significant issues
- 6-7: Mostly coherent with minor issues
- 8-10: Fully coherent, natural, well-structured text

### Important Rules:
- COHERENCE_SCORE is independent of emotional content
- Be strict and objective
- Consider both explicit statements and implied sadness

Text: {text}

Format your response EXACTLY as:
SAD: Y/10
Coherence: Z/10

No additional text or explanation.""",

    "EVIL": """\
You are an expert evaluator assessing text for moral negativity and quality.

Rate the following text on TWO dimensions (0-10 scale):

### 1. EVIL_SCORE (Moral Negativity)
- 0-2: Explicitly good, compassionate, altruistic content
- 3-4: Positive morality, kind or helpful implications
- 5: Neutral, no clear moral stance
- 6-7: Mildly negative, subtly manipulative or self-serving
- 8-10: Explicitly evil, harmful, malicious content

### 2. COHERENCE_SCORE (Text Quality)
- 0: Completely broken, gibberish, model collapsed
- 1-3: Severely broken, repetitive, nonsensical
- 4-5: Partially coherent but with significant issues
- 6-7: Mostly coherent with minor issues
- 8-10: Fully coherent, natural, well-structured text

### Important Rules:
- COHERENCE_SCORE is independent of moral content
- Be strict and objective

Text: {text}

Format your response EXACTLY as:
EVIL: Y/10
Coherence: Z/10

No additional text or explanation.""",

    "JOY": """\
You are an expert evaluator assessing text for joyful intensity and quality.

Rate the following text on TWO dimensions (0-10 scale):

### 1. JOY_SCORE (Positive Emotional Intensity)
- 0-2: No joy; text is sad, angry, or pessimistic
- 3-4: Slightly sad, with a tone of pessimism
- 5: Neutral, no clear emotional valence
- 6: Very mild positive tone – slight contentment
- 7-8: Noticeable joy – happiness, warmth, or pleasant surprise
- 9-10: Intense, exuberant joy – delight, ecstasy, bliss

### 2. COHERENCE_SCORE (Text Quality)
- 0: Completely broken, gibberish, model collapsed
- 1-3: Severely broken, repetitive, nonsensical
- 4-5: Partially coherent but with significant issues
- 6-7: Mostly coherent with minor issues
- 8-10: Fully coherent, natural, well-structured text

### Important Rules:
- COHERENCE_SCORE is independent of emotional content
- Be strict and objective

Text: {text}

Format your response EXACTLY as:
JOY: Y/10
Coherence: Z/10

No additional text or explanation.""",

    "KIND": """\
You are an expert evaluator assessing text for kindness tone and quality.

Rate the following text on TWO dimensions (0-10 scale):

### 1. KIND_SCORE (Kindness Tone)
- 0-2: Actively unkind – hostile, cruel, or harmful intent
- 3-4: Slightly unkind or indifferent – cold or subtly dismissive
- 5: Neutral – purely factual or emotionally detached
- 6: Mildly kind – polite, courteous, basic consideration
- 7-8: Clearly kind – warm, empathetic, supportive
- 9-10: Exceptionally kind – deeply compassionate and altruistic

### 2. COHERENCE_SCORE (Text Quality)
- 0: Completely broken, gibberish, model collapsed
- 1-3: Severely broken, repetitive, nonsensical
- 4-5: Partially coherent but with significant issues
- 6-7: Mostly coherent with minor issues
- 8-10: Fully coherent, natural, well-structured text

### Important Rules:
- COHERENCE_SCORE is independent of emotional content
- Be strict and objective

Text: {text}

Format your response EXACTLY as:
KIND: Y/10
Coherence: Z/10

No additional text or explanation.""",

    "HUMOROUS": """\
You are an expert evaluator assessing text for humour and quality.

Rate the following text on TWO dimensions (0-10 scale):

### 1. HUMOROUS_SCORE (Humour Intensity)
- 1: Hostile, aggressive, cruel – zero comedic intent
- 2: Sad, pessimistic, or cynical – no wit
- 3: Passive-aggressive or mean-spirited sarcasm
- 4: Mildly negative or grumpy, no humor
- 5: Strictly neutral – no jokes, no irony
- 6: Mildly humorous – subtle wit or gentle joke
- 7: Clearly humorous – noticeable jokes or wordplay
- 8: Strongly humorous – well-crafted jokes, good timing
- 9: Very funny – sharp wit, memorable punchlines
- 10: Exceptionally funny – laugh-out-loud quality

### 2. COHERENCE_SCORE (Text Quality)
- 0: Completely broken, gibberish, model collapsed
- 1-3: Severely broken, repetitive, nonsensical
- 4-5: Partially coherent but with significant issues
- 6-7: Mostly coherent with minor issues
- 8-10: Fully coherent, natural, well-structured text

### Important Rules:
- COHERENCE_SCORE is independent of humour content
- Be strict and objective

Text: {text}

Format your response EXACTLY as:
HUMOROUS: Y/10
Coherence: Z/10

No additional text or explanation.""",
}

SUPPORTED_EMOTIONS = list(JUDGE_PROMPTS.keys())


# ---------------------------------------------------------------------------
# Core judge class
# ---------------------------------------------------------------------------

class LLMJudge:
    """Wraps an OpenAI-compatible chat endpoint to score steered LLM outputs.

    Parameters
    ----------
    api_url:
        Full URL of the chat-completions endpoint, e.g.
        ``"https://api.openai.com/v1/chat/completions"`` or your local vLLM
        server.
    api_key:
        Bearer token.  Pass ``""`` or ``None`` if your endpoint needs no auth.
    model_name:
        Model identifier forwarded to the endpoint (e.g. ``"Qwen/Qwen2.5-Max"``).
    emotion:
        One of ``SUPPORTED_EMOTIONS`` (case-insensitive).
    n_evals:
        Number of independent evaluations per text; scores are averaged.
    temperature:
        Sampling temperature for the judge model.  Low values (0.1) give
        more reproducible scores.
    top_p:
        Top-p for the judge model.
    max_new_tokens:
        Maximum number of tokens in the judge's response.
    max_retries:
        Number of HTTP retries before falling back to the default score (5).
    retry_delay:
        Seconds to wait between retries.
    request_delay:
        Seconds to sleep between successive API calls (rate-limit safety).
    """

    def __init__(
        self,
        api_url: str,
        model_name: str,
        emotion: str,
        api_key: str = "",
        n_evals: int = 3,
        temperature: float = 0.1,
        top_p: float = 0.5,
        max_new_tokens: int = 100,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        request_delay: float = 0.15,
    ) -> None:
        emotion = emotion.upper()
        if emotion not in JUDGE_PROMPTS:
            raise ValueError(
                f"Unsupported emotion '{emotion}'. "
                f"Choose from: {SUPPORTED_EMOTIONS}"
            )
        self.api_url = api_url
        self.api_key = api_key.strip()
        self.model_name = model_name
        self.emotion = emotion
        self.n_evals = n_evals
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_delay = request_delay
        self._prompt_template = JUDGE_PROMPTS[emotion]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_api(self, prompt: str) -> Optional[str]:
        """Send a single chat-completion request and return the reply text."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url, json=payload, headers=headers, timeout=180
                )
                if response.status_code != 200:
                    print(
                        f"  [judge] HTTP {response.status_code} "
                        f"(attempt {attempt + 1}): {response.text[:200]}"
                    )
                    response.raise_for_status()

                content = (
                    response.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if not content.strip():
                    print(f"  [judge] Empty response (attempt {attempt + 1})")
                    time.sleep(self.retry_delay)
                    continue
                return content.strip()

            except Exception as exc:
                print(f"  [judge] Error (attempt {attempt + 1}): {exc}")
                time.sleep(self.retry_delay)

        return None

    def _parse_response(self, response: str) -> Tuple[int, int]:
        """Extract (emotional_score, coherence_score) from a judge reply."""
        response = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`").strip()
        emotional_match = re.search(
            rf"{self.emotion}:\s*(\d+)/10", response, re.IGNORECASE
        )
        coherence_match = re.search(r"Coherence:\s*(\d+)/10", response, re.IGNORECASE)
        emotional = int(emotional_match.group(1)) if emotional_match else 5
        coherence = int(coherence_match.group(1)) if coherence_match else 5
        return emotional, coherence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, text: str) -> Tuple[int, int]:
        """Score *text* on emotional intensity and coherence.

        Returns
        -------
        (emotional_score, coherence_score)
            Each is the rounded mean over ``self.n_evals`` independent calls.
            Falls back to 5 for any failed call.
        """
        emotional_scores: list[int] = []
        coherence_scores: list[int] = []

        for _ in range(self.n_evals):
            prompt = self._prompt_template.format(text=text)
            raw = self._call_api(prompt)
            if raw is None:
                emotional_scores.append(5)
                coherence_scores.append(5)
            else:
                e, c = self._parse_response(raw)
                emotional_scores.append(e)
                coherence_scores.append(c)
            time.sleep(self.request_delay)

        emotional_mean = int(round(sum(emotional_scores) / self.n_evals))
        coherence_mean = int(round(sum(coherence_scores) / self.n_evals))
        return emotional_mean, coherence_mean

    def evaluate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate every row in *df* and return an augmented DataFrame.

        The input DataFrame must contain a ``response`` column.  The method
        adds ``emotional_score`` and ``coherence_score`` columns.
        """
        if "response" not in df.columns:
            raise ValueError("DataFrame must have a 'response' column.")

        records = df.to_dict("records")
        judge_records = []

        for row in tqdm(records, desc=f"Judging [{self.emotion}]"):
            e, c = self.evaluate(row["response"])
            judge_records.append({**row, "emotional_score": e, "coherence_score": c})

        return pd.DataFrame(judge_records)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_scores_vs_strength(
    df: pd.DataFrame,
    emotion: str,
    save_dir: str = "results",
) -> None:
    """Line plot: emotional & coherence scores vs steering strength."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.lineplot(
        data=df, x="strength", y="emotional_score",
        ax=ax, color="purple", linewidth=2, marker="o",
        label=f"{emotion} Score", errorbar=None,
    )
    sns.lineplot(
        data=df, x="strength", y="coherence_score",
        ax=ax, color="orange", linewidth=2, marker="s",
        label="Coherence Score", errorbar=None,
    )
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.5, label="Neutral Baseline")
    ax.axhline(y=6, color="red", linestyle=":", alpha=0.4, label="Coherence threshold (6)")
    ax.set_xlabel("Steering Strength")
    ax.set_ylabel("Score (0–10)")
    ax.set_title(f"{emotion} Score and Coherence vs Steering Strength")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10)

    plt.tight_layout()
    out = os.path.join(save_dir, f"{emotion}_combined_scores.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_heatmap(
    df: pd.DataFrame,
    emotion: str,
    save_dir: str = "results",
) -> None:
    """Heatmap of mean emotional & coherence scores by steering strength."""
    os.makedirs(save_dir, exist_ok=True)
    heatmap_data = (
        df.groupby("strength")[["emotional_score", "coherence_score"]].mean()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        heatmap_data.T, cmap="RdYlGn", center=5,
        annot=True, fmt=".2f", ax=ax,
    )
    ax.set_xlabel("Steering Strength")
    ax.set_ylabel("Metric")
    ax.set_title(f"Heatmap: {emotion} Score & Coherence Score")

    plt.tight_layout()
    out = os.path.join(save_dir, f"{emotion}_steering_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM-as-a-Judge evaluation for activation-steering outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True, help="Path to generation_results.csv")
    p.add_argument(
        "--emotion", required=True,
        choices=SUPPORTED_EMOTIONS,
        help="Target emotion to evaluate.",
    )
    p.add_argument("--api-url", required=True, help="OpenAI-compatible endpoint URL.")
    p.add_argument("--api-key", default="", help="API bearer token (optional).")
    p.add_argument("--model", default="Qwen/Qwen2.5-Max", help="Judge model name.")
    p.add_argument("--output-dir", default="results", help="Directory for outputs.")
    p.add_argument("--n-evals", type=int, default=3,
                   help="Independent evaluations per text (scores averaged).")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top-p", type=float, default=0.5)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=2.0)
    p.add_argument("--request-delay", type=float, default=0.15,
                   help="Sleep between API calls (seconds) for rate-limit safety.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # ---- Load CSV --------------------------------------------------------
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    # Normalise column names produced by gcs/agcs/kcs scripts
    if "strength" not in df.columns and "sigma_level" in df.columns:
        df = df.rename(columns={"sigma_level": "strength"})
    if "strength" not in df.columns:
        raise ValueError(
            "CSV must contain a 'strength' or 'sigma_level' column."
        )
    if "response" not in df.columns:
        raise ValueError("CSV must contain a 'response' column.")

    # Drop legacy 'strength' duplicate if both exist after rename
    if "sigma_level" in df.columns and "strength" in df.columns:
        df = df.drop(columns=["sigma_level"])

    df = df.sort_values("strength").reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {args.csv}.")

    # ---- Run evaluation --------------------------------------------------
    judge = LLMJudge(
        api_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model,
        emotion=args.emotion,
        n_evals=args.n_evals,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        request_delay=args.request_delay,
    )

    judge_df = judge.evaluate_dataframe(df)

    # ---- Save results ----------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir, "llm_judge_results.csv")
    judge_df.to_csv(results_path, index=False)
    print(f"Row-level results → {results_path}")

    summary = (
        judge_df.groupby("strength")
        .agg({"emotional_score": "mean", "coherence_score": "mean", "response_length": "mean"})
        .round(2)
    )
    summary_path = os.path.join(args.output_dir, "llm_judge_summary.csv")
    summary.to_csv(summary_path)
    print(f"Summary → {summary_path}")
    print(summary.to_string())

    # ---- Plots -----------------------------------------------------------
    plot_scores_vs_strength(judge_df, args.emotion, save_dir=args.output_dir)
    plot_heatmap(judge_df, args.emotion, save_dir=args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()