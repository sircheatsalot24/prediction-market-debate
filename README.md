# Prediction Market Debate System

A multi-agent AI system that debates both sides of a live Polymarket prediction market question and delivers a reasoned verdict.

## How It Works

1. Fetches a random active market from Polymarket
2. A **bull agent** uses web search and market data tools to build a case for YES
3. A **bear agent** does the same for NO
4. A **judge agent** evaluates both arguments and delivers a verdict with reasoning

Each agent has access to two tools: live market data lookup and web search, so their arguments are grounded in real current information.

## Stack

- Python (async)
- OpenAI API (GPT-4o-mini + web search tool)
- Polymarket API

## Setup

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install openai polymarket python-dotenv
   ```
3. Create a `.env` file:
   ```
   OPENAI_API_KEY=your_key_here
   ```
4. Run:
   ```bash
   python predictionmarket.py
   ```

## Example Output

```
Will Michelle Obama win the 2028 Democratic presidential nomination?

bear response: The prediction market places the "No" probability at 98.65%...
bull response: Despite low market odds, Michelle Obama's popularity and influence...
judge's verdict: Winner: the bear analyst. Reasoning: The bear case relies on 
tangible current signals including explicit statements from Obama herself...
```