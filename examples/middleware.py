"""







"""
import asyncio
import os
from dataclasses import dataclass

from llmkit import Event, MiddlewareOp, MiddlewarePhase
from llmkit.builders import anthropic


@dataclass
class Price:
    input: float
    output: float


class SpendCap:
    def __init__(self, budget: float, prices: dict[str, Price]) -> None:
        self.budget = budget
        self.spent = 0.0
        self.prices = prices

    def middleware(self, e: Event) -> Exception | None:
        if e.op != MiddlewareOp.LLM_REQUEST:
            return None
        if e.phase == MiddlewarePhase.PRE:
            if self.spent >= self.budget:
                return RuntimeError(
                    f"daily budget ${self.budget:.2f} exceeded "
                    f"(spent ${self.spent:.4f})"
                )
            return None
        p = self.prices.get(e.model)
        if p is None or e.usage is None:
            return None
        self.spent += (
            e.usage.input * p.input / 1e6 + e.usage.output * p.output / 1e6
        )
        return None


def token_logger(e: Event) -> Exception | None:
    if (
        e.op == MiddlewareOp.LLM_REQUEST
        and e.phase == MiddlewarePhase.POST
        and e.usage is not None
    ):
        print(
            f"[{e.provider}/{e.model}] in={e.usage.input} out={e.usage.output} "
            f"cache_read={e.usage.cache_read} took={e.duration:.3f}s"
        )
    return None


async def main() -> None:
    cap = SpendCap(
        5.00,
        {"claude-sonnet-4-5-20250929": Price(input=3.00, output=15.00)},
    )
    c = anthropic(os.environ.get("ANTHROPIC_API_KEY", "sk-test"))
    resp = await (
        c.text
        .add_middleware(cap.middleware, token_logger)
        .prompt("What is 2+2? Reply in one word.")
    )
    print("Answer:", resp.text)
    print(f"Spent so far: ${cap.spent:.4f} / ${cap.budget:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
