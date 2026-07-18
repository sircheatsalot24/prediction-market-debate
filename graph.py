from typing import List, TypedDict, Literal, Dict, Union
from langchain.agents import create_agent
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from polymarket import AsyncPublicClient, Market
from langgraph.graph import END, START, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import asyncio, random, os

class AnalystResult(BaseModel):
    main_reason: str = Field(description="The main reason why the judge should support your side.")
    additional_reasoning: str = Field(description="Additional reasoning to support your analysis. Keep it to 5-6 sentences, and 8 sentences maximum.")
class Verdict(BaseModel):
    final_decision: Literal["BEAR", "BULL"] = Field(description="Your final decision. Should be either \"BEAR\" or \"BULL\"")
    reasoning: str = Field(description="Additional reasoning to support your decision. Keep it to 5-6 sentences, and 8 sentences maximum.")

class State(TypedDict):
    market_id: str
    keywords: List[str]
    random_market_chosen: Dict[str, Union[int, str]]
    bull_system_prompt: SystemMessage
    bear_system_prompt: SystemMessage
    bear_result: AnalystResult
    bull_result: AnalystResult
    judge_messages: list[BaseMessage]
    verdict: Verdict

load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@tool
async def search(question: str) -> list[dict]:
    """A search tool. Takes a question and returns web search outputs."""
    print("An agent used a tool: Search")
    response = await asyncio.to_thread(openai.responses.create,
        model="gpt-4o-mini",
        tools=[{"type": "web_search"}],
        input=question,
    )

    return [{"question": question}, {"search results": response.output_text}]

async def random_market(id: str = None, keywords: List[str] = []):

    async with AsyncPublicClient() as client:
        if id:
            chosen = await client.get_market(id=id)
        elif keywords:
            random.shuffle(keywords)
            cands: List[Market] = []
            found = False
            lower, upper = 0.35, 0.65
            for keyword in keywords:
                paginator = client.search(q=keyword)
                raw = await paginator.first_page()
                for item in raw.items:
                    for events in item.events:
                        for market in events.markets:
                            if market.state.active and lower < float(market.outcomes.yes.price) < upper and market.prices.last_trade_price is not None:
                                cands.append(market)

                if len(cands) != 0:
                    chosen = random.choice(cands)
                    found = True
                    break
                else:
                    continue

            if not found:
                raise LookupError(f"No active market found for all keywords: {keywords}")


        else:
            markets = client.list_markets(closed=False, page_size=100)
            first_page = await markets.first_page()
            items = first_page.items
            chosen = random.choice(items)
        print(f"Question: {chosen.question}")
    return {
        "id": chosen.id,
        "question": chosen.question,
        "description": chosen.description,
        "temp":  {
                    "market question": chosen.question,
                    "market description": chosen.description,
                    "market prices": {
                        "last trade price": float(chosen.prices.last_trade_price),
                        "one week change": float(chosen.prices.one_week_price_change or 0),
                        "one month change": float(chosen.prices.one_month_price_change or 0),
                        "one year change": float(chosen.prices.one_year_price_change or 0),
                    },
                    "market probabilities": {
                        "yes price": float(chosen.outcomes.yes.price),
                        "no price": float(chosen.outcomes.no.price),
                    },
                }}

@tool
async def market_lookup(market_id: str):
    """A market lookup tool. Takes a market_id and returns values of the market"""
    print("An agent used a tool: Market Lookup")
    async with AsyncPublicClient() as client:
        market = await client.get_market(id=market_id)
        temp = {
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
        return temp

tools = [market_lookup, search]


async def setup(state: State) -> State:

    if not state.get("random_market_chosen"):
        if state.get("market_id"):
            state["random_market_chosen"] = await random_market(id=state["market_id"])
        elif state.get("keywords"):
            state["random_market_chosen"] = await random_market(keywords=state["keywords"])
        else:
            state["random_market_chosen"] = await random_market()

    state["bull_system_prompt"] = SystemMessage(content=f"""
    You are a bull analyst arguing that the answer to this prediction market is YES.
    Use your market_lookup and search tools to find evidence. Be persuasive and specific.
    You MUST USE at least one instance of the following tools: {"\n".join(f"- {t.name}: {t.description}" for t in tools)}
        MAKE SURE TO USE ALL OF YOUR TOOLS AT LEAST ONCE.
    """)

    state["bear_system_prompt"] = SystemMessage(content=f"""
    You are a bear analyst arguing that the answer to this prediction market is NO.
    Use your market_lookup and search tools to find evidence. Be persuasive and specific.
    You MUST USE at least one instance of the following tools: {"\n".join(f"- {t.name}: {t.description}" for t in tools)}
    MAKE SURE TO USE ALL OF YOUR TOOLS AT LEAST ONCE.
    """)

    return state


async def bull(state: State) -> Dict:

    llm = create_agent(
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools=tools,
        system_prompt=state["bull_system_prompt"],
        response_format=AnalystResult,
    )

    response = await llm.ainvoke({"messages": HumanMessage(content=f"Here is the market ID: {state["random_market_chosen"]["id"]}.")})

    response = response["structured_response"]

    cleaned = f"main reason: {response.main_reason}\nreasoning: {response.additional_reasoning}"

    if cleaned:
        print("Bull responded!")
    return {"bull_result": response}

async def bear(state: State) -> Dict:

    llm = create_agent(
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools=tools,
        system_prompt=state["bear_system_prompt"],
        response_format=AnalystResult,
    )

    response = await llm.ainvoke({"messages": HumanMessage(content=f"Here is the market ID: {state["random_market_chosen"]["id"]}.")})

    response = response["structured_response"]

    cleaned = f"main reason: {response.main_reason}\nreasoning: {response.additional_reasoning}"

    if cleaned:
        print("Bear responded!")
    return {"bear_result": response}


def judge(state: State) -> Dict:
    random_market_chosen = state["random_market_chosen"]
    state["judge_messages"] = [SystemMessage(content="""
    You are an impartial judge evaluating two analysts' arguments about a prediction market question. 
    You will receive a bull case (arguing YES) and a bear case (arguing NO). 
    Evaluate the quality of evidence, logical consistency, and persuasiveness of each argument. 
    Declare a winner and explain your reasoning in 5-8 sentences.
    """),
    HumanMessage(content=f"""
    Here is the bull analyst's response: \n    "Main Reason: {state["bull_result"].main_reason}\n    Additional Reasoning: {state["bull_result"].additional_reasoning}"\n\n
    Here is the bear analyst's response: \n    "Main Reason: {state["bear_result"].main_reason}\n    Additional Reasoning: {state["bear_result"].additional_reasoning}"\n\n\n
    Here is the question given: \n    "{random_market_chosen["question"]}"\n\n
    Here is the question's description: \n    "{random_market_chosen["description"]}"
    """
    )]


    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Verdict)
    result = judge.invoke(state["judge_messages"])
    print(f"Judge responded! Winner: {result.final_decision}")
    return {"verdict": result}

graph = StateGraph(State)

graph.add_node("setup", setup)
graph.add_node("bull", bull)
graph.add_node("bear", bear)
graph.add_node("judge", judge)

graph.add_edge(START, "setup")
graph.add_edge("judge", END)

graph.add_edge("setup", "bull")
graph.add_edge("setup", "bear")

graph.add_edge("bull", "judge")
graph.add_edge("bear", "judge")

compiled = graph.compile()

def run_graph(args_dict):
    return asyncio.run(compiled.ainvoke(args_dict))

async def arun_graph(args_dict):
    return await compiled.ainvoke(args_dict)