from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Literal
from . import gemini_runtime, mock_runtime
from .schemas import AttemptTrace, JudgeResult, QAExample, ReflectionEntry, RunRecord

def _token_count(response: object) -> int:
    usage = getattr(response, "usage", None)
    if usage is not None:
        for key in ("total_tokens", "total_token_count", "completion_tokens", "output_tokens"):
            value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
            if isinstance(value, int):
                return value
    if hasattr(response, "model_dump_json"):
        return len(response.model_dump_json().split())
    return len(str(response).split())

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    runtime: Literal["mock", "gemini"] = "mock"
    model: str = gemini_runtime.DEFAULT_GEMINI_MODEL

    def _actor_answer(self, example: QAExample, attempt_id: int, reflection_memory: list[str]) -> tuple[str, int]:
        if self.runtime == "gemini":
            return gemini_runtime.actor_answer(example, attempt_id, self.agent_type, reflection_memory, model=self.model)
        answer = mock_runtime.actor_answer(example, attempt_id, self.agent_type, reflection_memory)
        return answer, _token_count(answer)

    def _evaluator(self, example: QAExample, answer: str) -> tuple[JudgeResult, int]:
        if self.runtime == "gemini":
            return gemini_runtime.evaluator(example, answer, model=self.model)
        judge = mock_runtime.evaluator(example, answer)
        return judge, _token_count(judge)

    def _reflector(self, example: QAExample, attempt_id: int, judge: JudgeResult) -> tuple[ReflectionEntry, int]:
        if self.runtime == "gemini":
            return gemini_runtime.reflector(example, attempt_id, judge, model=self.model)
        reflection = mock_runtime.reflector(example, attempt_id, judge)
        return reflection, _token_count(reflection)

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            attempt_start_ns = perf_counter_ns()
            answer, answer_tokens = self._actor_answer(example, attempt_id, reflection_memory)
            judge, judge_tokens = self._evaluator(example, answer)
            token_estimate = answer_tokens + judge_tokens
            reflection = None
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                latency_ms = (perf_counter_ns() - attempt_start_ns) // 1_000_000
                trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
                traces.append(trace)    
                break
            
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection, reflection_tokens = self._reflector(example, attempt_id, judge)
                token_estimate += reflection_tokens
                reflections.append(reflection)
                reflection_memory.append(
                    f"Attempt {attempt_id} failed: {reflection.failure_reason} "
                    f"Lesson: {reflection.lesson} "
                    f"Next strategy: {reflection.next_strategy}"
                )
            latency_ms = (perf_counter_ns() - attempt_start_ns) // 1_000_000
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, reflection=reflection, token_estimate=token_estimate, latency_ms=latency_ms)
            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else mock_runtime.FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self, runtime: Literal["mock", "gemini"] = "mock", model: str = gemini_runtime.DEFAULT_GEMINI_MODEL) -> None:
        super().__init__(agent_type="react", max_attempts=1, runtime=runtime, model=model)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, runtime: Literal["mock", "gemini"] = "mock", model: str = gemini_runtime.DEFAULT_GEMINI_MODEL) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, runtime=runtime, model=model)
