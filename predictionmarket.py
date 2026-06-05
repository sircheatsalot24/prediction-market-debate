from openai import OpenAI
import os
from dotenv import load_dotenv
from polymarket import AsyncPublicClient
import asyncio
import random
import json

load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def search(question: str) -> list[dict]:
    response = await asyncio.to_thread(openai.responses.create,
        model="gpt-4o-mini",
        tools=[{"type": "web_search"}],
        input=question,
    )

    print("used tool: search")
    return [{"question": question}, {"search results": response.output_text}]

async def random_market():
    async with AsyncPublicClient() as client:
        markets = client.list_markets(closed=False, page_size=100)
        first_page = await markets.first_page()
        items = first_page.items
        chosen = random.choice(items)
        print(chosen.question, chosen.description)
    return {"id": chosen.id, "question": chosen.question, "description": chosen.description}



async def market_lookup(market_id: str):
    print("used tool: market lookup")
    async with AsyncPublicClient() as client:
        market = await client.get_market(id=market_id)
        return {
                    "market question": market.question, 
                    "market description": market.description, 
                    "market prices": {
                        "last trade price": float(market.prices.last_trade_price),
                        "one week change": float(market.prices.one_week_price_change or 0),
                        "one month change": float(market.prices.one_month_price_change or 0),
                        "one year change": float(market.prices.one_year_price_change or 0),
                    },
                    "market probabilities": {
                        "yes price": float(market.outcomes.yes.price),
                        "no price": float(market.outcomes.no.price),
                    },
                }



search_json = {
    "name": "search",
    "description": "Search the web for recent news and information",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string"}
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

market_lookup_json = {
    "name": "market_lookup",
    "description": "Look up Market information by Market id",
    "parameters": {
        "type": "object",
        "properties": {
            "market_id": {"type": "string"}
        },
        "required": ["market_id"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": search_json},
    {"type": "function", "function": market_lookup_json},
]

bull_system_prompt = """
You are a bull analyst arguing that the answer to this prediction market is YES. 
Use your market_lookup and search tools to find evidence. Be persuasive and specific.
"""

bear_system_prompt = """
You are a bear analyst arguing that the answer to this prediction market is NO.
Use your market_lookup and search tools to find evidence. Be persuasive and specific.
"""

bear_messages = [
    {"role": "system", "content": bear_system_prompt}
]

bull_messages = [
    {"role": "system", "content": bull_system_prompt},
]

async def bull():
    response = await asyncio.to_thread(openai.chat.completions.create,
    model="gpt-4o-mini",
    messages=bull_messages,
    tools=tools
    )

    cleaned = response.choices[0].message.content

    bull_messages.append(response.choices[0].message)
    if cleaned:
        print(f"\n\n\n\n\n\n\nbull response: {cleaned}")
    return response

async def bear():
    response = await asyncio.to_thread(openai.chat.completions.create, 
    model="gpt-4o-mini",
    messages=bear_messages,
    tools=tools
    )

    cleaned = response.choices[0].message.content

    bear_messages.append(response.choices[0].message)
    if cleaned:
        print(f"\n\n\n\n\n\n\nbear response: {cleaned}")
    return response


async def main():
    random_market_chosen = await random_market()
    random_market_id = random_market_chosen["id"]
    bull_messages.append({"role": "user", "content": f"Here is the market ID: {random_market_id}."})
    bear_messages.append({"role": "user", "content": f"Here is the market ID: {random_market_id}."})

    async def loop():
        async def run_bull():
            while True:
                bull_message = await bull()

                if bull_message.choices[0].finish_reason == "tool_calls":
                    for tool_call in bull_message.choices[0].message.tool_calls:
                        result = await globals().get(tool_call.function.name)(**json.loads(tool_call.function.arguments))
                        bull_messages.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
                elif bull_message.choices[0].finish_reason == "stop":
                    break
            
            return bull_message.choices[0].message.content

        async def run_bear():
            while True:
                bear_message = await bear()

                if bear_message.choices[0].finish_reason == "tool_calls":
                    for tool_call in bear_message.choices[0].message.tool_calls:
                        result = await globals().get(tool_call.function.name)(**json.loads(tool_call.function.arguments))
                        bear_messages.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
                elif bear_message.choices[0].finish_reason == "stop":
                    break
            
            return bear_message.choices[0].message.content

        bull_result, bear_result = await asyncio.gather(run_bull(), run_bear())

        return {"bull": bull_result, "bear": bear_result}
    

    judge_system_prompt = """
    You are an impartial judge evaluating two analysts' arguments about a prediction market question. 
    You will receive a bull case (arguing YES) and a bear case (arguing NO). 
    Evaluate the quality of evidence, logical consistency, and persuasiveness of each argument. 
    Declare a winner and explain your reasoning in 5-8 sentences.
    """

    results = await loop()
    bear_result = results["bear"]
    bull_result = results["bull"]

    judge = await asyncio.to_thread(openai.chat.completions.create,
    model="gpt-5-nano",
    messages=[
        {"role": "system", "content": judge_system_prompt}, 
        {"role": "user", "content": f"Here is the bull analyst's response: \"{bull_result}\"\n\n\n\n\nHere is the bear analyst's response: \"{bear_result}\"\n\n\n\n\nHere is the question given: \"{random_market_chosen["question"]}\"\n\n\n\n\nHere is the question's description: \"{random_market_chosen["description"]}\""}
        ]
    )

    print(f"\n\n\n\n\n\n\njudge's verdict: {judge.choices[0].message.content}")

asyncio.run(main())