from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, TypedDict, Union
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from graph import AnalystResult, Verdict, random_market, run_graph, arun_graph
import asyncio, random

@dataclass
class GraphExecution:
    bull_result: AnalystResult
    bear_result: AnalystResult
    judge_result: Verdict
    market_info: Dict[str, Union[str, Dict[str, Union[float, int]]]]

class EvaluationCriteria(BaseModel):
    total: int = Field(
        ge=0, le=100,
        description="Sum of all criteria scores for this evaluation, out of 100."
    )
    feedback: str = Field(
        description="2-4 sentences of specific, actionable feedback explaining the scores above — "
                     "what was strong, what was weak, and why."
    )
    consistency: int = Field(
        ge=0, le=20,
        description="Does the reasoning hang together internally — does the supporting detail actually "
                     "support the stated main conclusion, without contradiction or non-sequitur?"
    )
    phrasing: int = Field(
        ge=0, le=20,
        description="Is the writing clear, precise, and free of vague hedging or filler? Penalize "
                     "generic boilerplate that could apply to almost any market."
    )


class AnalystEvaluation(EvaluationCriteria):
    evidence_use: int = Field(
        ge=0, le=20,
        description="Does the argument use market data, search results, or other evidence to support "
                     "independent analytical reasoning — rather than treating a single data point (e.g. "
                     "the market's current price) as the entire justification on its own?"
    )
    persuasiveness: int = Field(
        ge=0, le=20,
        description="How compelling and well-supported is the case, independent of which side it argues?"
    )
    relevance: int = Field(
        ge=0, le=20,
        description="Is the argument specific to this market's actual question and details, rather than "
                     "generic reasoning that could apply to almost any similar market?"
    )


class JudgeEvaluation(EvaluationCriteria):
    identification: int = Field(
        ge=0, le=20,
        description="Did the judge correctly identify which side (bull or bear) presented the stronger "
                     "argument, based on evidence quality rather than which side happened to go first?"
    )
    reasoning_traceability: int = Field(
        ge=0, le=20,
        description="Does the judge's stated reasoning actually reference specific claims made by bull "
                     "and bear, rather than generic post-hoc justification?"
    )
    decision_alignment: int = Field(
        ge=0, le=20,
        description="Does the final BEAR/BULL decision logically follow from the reasoning given, with "
                     "no mismatch between the stated reasoning and the declared winner?"
    )

@dataclass
class CompiledEvaluation:
    bear_evaluation: AnalystEvaluation
    bull_evaluation: AnalystEvaluation
    judge_evaluation: JudgeEvaluation

keywords = [
    # Sports
    "nba", "nfl", "mlb", "nhl", "soccer", "premier league", "champions league",
    "tennis", "golf", "f1", "ufc", "boxing", "olympics", "world cup",
    "college football", "cricket",
    # Crypto/finance
    "bitcoin", "ethereum", "solana", "crypto", "stocks", "s&p 500", "fed",
    "interest rates", "inflation", "recession", "ipo",
    # Politics
    "election", "trump", "congress", "senate", "supreme court", "governor",
    "primary", "impeachment",
    # Entertainment/culture
    "movies", "box office", "oscars", "grammys", "music", "album", "netflix",
    "taylor swift", "celebrity",
    # Tech/science
    "ai", "openai", "spacex", "nasa", "apple", "tesla", "google", "meta",
    "chip", "iphone",
    # World events
    "ukraine", "china", "israel", "nato", "climate", "hurricane", "pandemic", "oil",
    # Miscellaneous
    "weather", "temperature", "time person of the year", "pope", "royal family",
]

def evaluate(mode: str, result, market_info):
    eval_llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0).with_structured_output(AnalystEvaluation if mode == "analyst" else JudgeEvaluation)
    eval = eval_llm.invoke([SystemMessage(content = analyst_eval_system_prompt if mode == "analyst" else judge_eval_system_prompt), HumanMessage(content = f"Here is the {"analyst" if mode == "analyst" else "judge"}'s result:\n{result}\nHere is the market information: {market_info}")])
    return eval

async def generate_testing_markets(amount):
    testing_markets = []
    seen_ids = set()
    while len(testing_markets) < amount:
        market = await random_market(keywords=keywords)
        if market["id"] not in seen_ids:
            seen_ids.add(market["id"])
            testing_markets.append(market)
        else:
            print(f"Duplicate draw, retrying ({len(testing_markets)}/{amount} so far)")

    return testing_markets


analyst_eval_system_prompt = """
    You are an impartial evaluator scoring an AI analyst's argument in a prediction-market debate system.

    You will be given the analyst's stated position (bull=YES or bear=NO), their main reason, their
    additional reasoning, and the market's question and description. Score the argument against five
    criteria, each from 0-20 points:

    - consistency: does the supporting detail actually support the stated main conclusion, without
    contradiction or non-sequitur?
    - phrasing: is the writing clear, precise, and free of vague hedging or generic filler?
    - evidence_use: does the argument use market data, search results, or other evidence to support
    independent analytical reasoning — rather than treating a single data point (e.g. the market's
    current price or a stated consensus) as the entire justification on its own?
    - persuasiveness: how compelling and well-supported is the case, independent of which side it argues?
    - relevance: is the argument specific to this market's actual question and details, rather than
    generic reasoning that could apply to almost any similar market?

    Be strict and consistent across evaluations — apply the same standard whether scoring a bull or a
    bear argument, and do not let confidence or lively phrasing substitute for actual evidentiary support.
    Provide the sum of your five scores as "total", and 2-4 sentences of specific, actionable feedback
    explaining what was strong, what was weak, and why.
"""

judge_eval_system_prompt = """
    You are an impartial evaluator scoring an AI judge's verdict in a prediction-market debate system.

    You will be given the judge's final decision (BEAR or BULL), their reasoning, and the original bull
    and bear arguments they were evaluating. Score the verdict against five criteria, each from 0-20 points:

    - consistency: does the supporting detail actually support the stated main conclusion, without
    contradiction or non-sequitur?
    - phrasing: is the writing clear, precise, and free of vague hedging or generic filler?
    - identification: did the judge correctly identify which side presented the stronger argument, based
    on evidence quality rather than which side happened to be presented first?
    - reasoning_traceability: does the judge's stated reasoning actually reference specific claims made by
    bull and bear, rather than generic post-hoc justification that could apply to any debate?
    - decision_alignment: does the final BEAR/BULL decision logically follow from the reasoning given, with
    no mismatch between the stated reasoning and the declared winner?

    Be strict and consistent across evaluations. Watch specifically for order bias — a judge that favors
    whichever argument happened to be presented first, rather than genuinely weighing evidence quality,
    should score low on identification regardless of which side it ultimately picked.
    Provide the sum of your five scores as "total", and 2-4 sentences of specific, actionable feedback
    explaining what was strong, what was weak, and why.
"""


async def main():
    print("Starting eval cycle!")
    testing_markets = await generate_testing_markets(int(input("Enter how many graph instances you would like:\n> ")))

    stored_results: Dict[str, GraphExecution] = {}

    completed = 0

    async def tracked_run(args):
        nonlocal completed
        result = await arun_graph(args)
        completed += 1
        print(f"Completed graph execution #{completed} of {len(testing_markets)}!")
        return result

    coroutine_list = [tracked_run({"random_market_chosen": market}) for market in testing_markets]

    res = await asyncio.gather(*coroutine_list)
    for market, result in zip(testing_markets, res):
        stored_results[market["id"]] = GraphExecution(
            bull_result=result["bull_result"],
            bear_result=result["bear_result"],
            judge_result=result["verdict"],
            market_info=result["random_market_chosen"]["temp"],
        )

    print("Successfully collected all graph executions!")

    compiled_evaluations: List[CompiledEvaluation] = []

    for result in stored_results.values():
        compiled_eval = CompiledEvaluation(
            bear_evaluation = evaluate("analyst", result.bear_result, result.market_info),
            bull_evaluation = evaluate("analyst", result.bull_result, result.market_info),
            judge_evaluation = evaluate("judge", {"judge_result": result.judge_result, "bear_result": result.bear_result, "bull_result": result.bull_result}, result.market_info),
        )

        compiled_evaluations.append(compiled_eval)

    print("Successfully built all compiled evaluations!")
    print("Here are the stats:")

    bear_addedup_dict, bull_addedup_dict, judge_addedup_dict = defaultdict(int), defaultdict(int), defaultdict(int)
    bear_counter, bull_counter, judge_counter = 0, 0, 0
    bear_feedback, bull_feedback, judge_feedback = [], [], []

    for eval in compiled_evaluations:
        bear_data = eval.bear_evaluation.model_dump(); bear_feedback.append(bear_data["feedback"]); del bear_data["feedback"]
        for key, value in bear_data.items():
            bear_addedup_dict[key] += value
        bear_counter += 1

        bull_data = eval.bull_evaluation.model_dump(); bull_feedback.append(bull_data["feedback"]); del bull_data["feedback"]
        for key, value in bull_data.items():
            bull_addedup_dict[key] += value
        bull_counter += 1

        judge_data = eval.judge_evaluation.model_dump(); judge_feedback.append(judge_data["feedback"]); del judge_data["feedback"]
        for key, value in judge_data.items():
            judge_addedup_dict[key] += value
        judge_counter += 1

    final_list = []

    for key in bear_addedup_dict:
        x = (f"Average Bear {key}: {round(bear_addedup_dict[key] / bear_counter, 3)}")
        final_list.append(x)
        print(x)

    for key in bull_addedup_dict:
        x = (f"Average Bull {key}: {round(bull_addedup_dict[key] / bull_counter, 3)}")
        final_list.append(x)
        print(x)

    for key in judge_addedup_dict:
        x = (f"Average Judge {key}: {round(judge_addedup_dict[key] / judge_counter, 3)}")
        final_list.append(x)
        print(x)

    final_list.extend([{"bear_feedback": bear_feedback, "bull_feedback": bull_feedback, "judge_feedback": judge_feedback}])

    print("Successfully interpreted evaluation data!")

    summarizer = ChatOpenAI(model = "gpt-4o-mini", temperature=0.5)
    messages = [SystemMessage(content = "You are a debate statistic summarizer. You take in feedback and stats from a finished, evaluated debate, and summarize them for a user to read, in order to understand the results of the debate. The user will see all the average total and criteria scores, so do not repeat them, as your main goal should be to provide feedback insight. DO NOT RESPOND IN MARKDOWN. RAW TEXT ONLY."), HumanMessage(f"Here are the stats from the debate:\n{final_list}")]
    for chunk in summarizer.stream(messages):
        print(chunk.content, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())