import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    ToolError,
    WorkerOptions,
    cli,
    function_tool,
    metrics,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

LEARNING_MODES = ("learn", "quiz", "teach_back")

VOICE_PERSONAS = {
    "learn": {
        "voice": "en-US-matthew",   # Required: Matthew
        "style": "Conversation",
        "display": "Matthew",
        "tone": "calm, encouraging explanations",
    },
    "quiz": {
        "voice": "en-US-alicia",    # Required: Alicia
        "style": "Conversation",
        "display": "Alicia",
        "tone": "energetic quiz master",
    },
    "teach_back": {
        "voice": "en-US-ken",       # Required: Ken
        "style": "Conversation",
        "display": "Ken",
        "tone": "supportive coach who listens closely",
    },
}


TUTOR_CONCEPTS_DATA = [
    {
        "id": "variables",
        "title": "Variables",
        "summary": "Variables store values so you can reuse them later, much like a labeled box you put information into. For example, if you store the number 10 in a variable named 'age', you can refer to that value simply by saying 'age'. This is essential for writing code that can adapt and remember information.",
        "sample_question": "What is a variable and why is it useful? Focus on the reusability aspect.",
        "teach_back_prompt": "Explain what a variable is and why it's useful in your own words."
    },
    {
        "id": "loops",
        "title": "Loops",
        "summary": "Loops let you repeat an action multiple times without having to write the same code over and over. Think of it like setting an alarm to go off every morning at 7 a.m.—the loop keeps repeating the action. The two main types are 'for' loops, which run a set number of times, and 'while' loops, which run as long as a certain condition is true.",
        "sample_question": "What is the difference between a for loop and a while loop?",
        "teach_back_prompt": "Explain the difference between a for loop and a while loop in detail."
    },
    {
        "id": "function",
        "title": "Functions",
        "summary": "Functions are blocks of organized, reusable code that perform a single, related action. They allow you to modularize your code, making it easier to read, test, and debug. When you need to perform an action multiple times, you simply call the function instead of writing the code repeatedly.",
        "sample_question": "Explain how functions help with code organization and reusability.",
        "teach_back_prompt": "Teach me back the concept of a function and its main benefits."
    },
    {
        "id": "if_else",
        "title": "If-Else Statements",
        "summary": "If-Else statements are the fundamental way to control the flow of a program. They allow your code to make decisions based on whether a condition is true or false. If the condition is true, the code in the 'if' block runs; otherwise, the code in the 'else' block runs.",
        "sample_question": "Describe a real-world scenario where an If-Else statement would be necessary in a program.",
        "teach_back_prompt": "Explain how if-else statements control program flow and give an example."
    },
    {
        "id": "data_types",
        "title": "Data Types",
        "summary": "Data types define the kind of value a variable can hold, such as numbers, text, or boolean (true/false) values. Common types include integers, floats, strings, and booleans. Using the correct data type is crucial for performing accurate operations and managing memory efficiently.",
        "sample_question": "What is a Data Type and what's the difference between an integer and a string?",
        "teach_back_prompt": "Teach me the difference between an integer, a string, and a boolean."
    },
    {
        "id": "operators",
        "title": "Operators",
        "summary": "Operators are special symbols that perform operations on variables and values. They are categorized into arithmetic (like +, -), comparison (like ==, >), and logical (like AND, OR) operators. They are the tools you use to manipulate data and create conditions in your code.",
        "sample_question": "Explain the difference between the assignment operator (=) and the comparison operator (==).",
        "teach_back_prompt": "Explain the three main categories of operators and give an example of one."
    },
    {
        "id": "oop",
        "title": "OOP (Object-Oriented Programming)",
        "summary": "Object-Oriented Programming is a paradigm based on the concept of 'objects,' which can contain data and code. The main principles are encapsulation, inheritance, and polymorphism. It helps manage complexity by modeling real-world entities and their interactions in code.",
        "sample_question": "Summarize the core principles of Object-Oriented Programming (OOP).",
        "teach_back_prompt": "Explain the core idea behind Object-Oriented Programming."
    }
]


@dataclass
class TutorConcept:
    """Structured representation of one concept."""

    id: str
    title: str
    summary: str
    sample_question: str
    teach_back_prompt: str


@dataclass
class ConceptMastery:
    """Simple counters that let the tutor track progress."""

    times_learned: int = 0
    times_quizzed: int = 0
    times_taught_back: int = 0
    last_score: Optional[int] = None
    last_feedback: Optional[str] = None


@dataclass
class TutorSessionState:
    """Conversation-specific session state."""

    current_mode: Optional[str] = None
    current_concept_id: Optional[str] = None
    mastery: Dict[str, ConceptMastery] = field(default_factory=dict)

    def ensure_mastery(self, concept_id: str) -> ConceptMastery:
        if concept_id not in self.mastery:
            self.mastery[concept_id] = ConceptMastery()
        return self.mastery[concept_id]


class TutorContentLibrary:
    """Loads and serves concept content from in-memory data."""

    def _init_(self, concepts: List[TutorConcept]):
        if not concepts:
            raise ValueError("TutorContentLibrary requires at least one concept.")
        self._concepts: Dict[str, TutorConcept] = {c.id: c for c in concepts}
        self._order: List[str] = [c.id for c in concepts]

    @classmethod
    def from_data(cls, data: List[Dict]) -> "TutorContentLibrary":
        """Loads content from the hardcoded list of dictionaries."""
        concepts = [TutorConcept(**item) for item in data]
        return cls(concepts)

    @classmethod
    def from_env(cls) -> "TutorContentLibrary":
        """Loads content from the global TUTOR_CONCEPTS_DATA."""
        global TUTOR_CONCEPTS_DATA
        return cls.from_data(TUTOR_CONCEPTS_DATA)
    
    def list_concepts(self) -> List[TutorConcept]:
        return [self._concepts[cid] for cid in self._order]

    def get(self, concept_id: Optional[str]) -> TutorConcept:
        target_id = concept_id or self._order[0]
        if target_id not in self._concepts:
            normalized_id = target_id.lower().replace(' ', '').replace('-', '')
            if normalized_id.endswith('s'):
                normalized_id = normalized_id[:-1]
                if normalized_id in self._concepts:
                     return self._concepts[normalized_id]
            raise KeyError(f"Unknown concept id: {target_id}")
        return self._concepts[target_id]

    def next_concept_id(self, current_id: Optional[str]) -> str:
        if current_id is None:
            return self._order[0]
        try:
            idx = self._order.index(current_id)
        except ValueError:
            return self._order[0]
        return self._order[(idx + 1) % len(self._order)]


@dataclass
class Userdata:
    """Holds both the session state and the content library."""

    state: TutorSessionState
    content: TutorContentLibrary


class TeachTheTutorAgent(Agent):
    """Active recall coach with mode-specific personas."""

    def _init_(self, *, userdata: Userdata) -> None:
        
        # --- CRITICAL INSTRUCTION FIX ---
        instructions = f"""You are Oracle, an active recall coach that helps users master core coding concepts.
Key behaviors:
- On the very first user message, if the user specifies a mode and a concept (e.g., 'start in quiz mode for loops'), *YOU MUST SKIP THE GENERAL GREETING* and *IMMEDIATELY* call the necessary tools (set_learning_mode, set_focus_concept) to fulfill their request.
- The default current concept is *variables*.
- Whenever the learner asks to switch mode OR concept, *IMMEDIATELY* call the set_learning_mode tool or set_focus_concept tool.
- When the mode changes, use the output of set_learning_mode to introduce the new voice (Matthew for learn, Alicia for quiz, Ken for teach_back) and then call the corresponding content tool (describe_current_concept, get_quiz_prompt, or get_teach_back_prompt).
- When the user answers a quiz or teach-back prompt, give brief, positive, qualitative feedback.
- Keep responses concise, use plain conversational language, and explain any jargon.
- Always mention that Murf Falcon provides the fast voices powering the experience at least once per conversation.

Concepts available for the set_focus_concept tool: {', '.join(userdata.content._order)}
"""
        super()._init_(instructions=instructions)

    def _apply_voice_persona(self, ctx: RunContext[Userdata], mode: str) -> None:
        persona = VOICE_PERSONAS[mode]
        tts_engine = ctx.session.tts
        if not tts_engine:
            logger.warning("Cannot switch voices: session has no TTS engine configured.")
            return

        update_cb = getattr(tts_engine, "update_options", None)
        if not callable(update_cb):
            logger.warning(
                "TTS engine %s does not support dynamic voice switching. Relying on LLM instruction.",
                getattr(tts_engine, "provider", "unknown"),
            )
            return

        try:
            update_cb(voice=persona["voice"], style=persona["style"])
            logger.info("Switched Murf voice to %s for %s mode.", persona["display"], mode)
        except Exception as exc: 
            logger.error("Failed to update TTS voice for mode %s: %s", mode, exc)

    @function_tool
    async def list_concepts(self, ctx: RunContext[Userdata]) -> str:
        """List available concepts with their IDs and titles so the learner can choose."""
        concepts = ctx.userdata.content.list_concepts()
        formatted = ", ".join(f"'{c.id}' ({c.title})" for c in concepts)
        return f"Available concepts: {formatted}. Ask the learner which one they want to focus on."

    @function_tool
    async def set_focus_concept(self, ctx: RunContext[Userdata], concept_id: str) -> str:
        """Set the active concept that the session should focus on."""
        concept = ctx.userdata.content.get(concept_id) 
        ctx.userdata.state.current_concept_id = concept.id
        ctx.userdata.state.ensure_mastery(concept.id)
        return f"Concept locked: {concept.title}. You're clear to continue working on {concept.title}."

    @function_tool
    async def describe_current_concept(self, ctx: RunContext[Userdata]) -> str:
        """Return the summary of the current concept for learn mode explanations."""
        concept = self._require_concept(ctx)
        mastery = ctx.userdata.state.ensure_mastery(concept.id)
        mastery.times_learned += 1
        return f"The concept is {concept.title}. Summary: {concept.summary}"

    @function_tool
    async def get_quiz_prompt(self, ctx: RunContext[Userdata]) -> str:
        """Return a quiz question for the current concept."""
        concept = self._require_concept(ctx)
        mastery = ctx.userdata.state.ensure_mastery(concept.id)
        mastery.times_quizzed += 1
        return f"Quiz question for {concept.title}: {concept.sample_question}"

    @function_tool
    async def get_teach_back_prompt(self, ctx: RunContext[Userdata]) -> str:
        """Return the teach-back instructions for the current concept."""
        concept = self._require_concept(ctx)
        mastery = ctx.userdata.state.ensure_mastery(concept.id)
        mastery.times_taught_back += 1
        return f"Teach-back prompt for {concept.title}: {concept.teach_back_prompt}"
    
    @function_tool
    async def set_learning_mode(self, ctx: RunContext[Userdata], mode: str) -> str:
        """Switch to one of the supported modes: learn, quiz, teach_back."""
        normalized = mode.lower()
        if normalized not in LEARNING_MODES:
            raise ToolError(
                f"Unsupported mode '{mode}'. Choose from: {', '.join(LEARNING_MODES)}."
            )
        ctx.userdata.state.current_mode = normalized
        persona = VOICE_PERSONAS[normalized]
        self._apply_voice_persona(ctx, normalized) 

        # The LLM will now receive this output and use its instructions to call the next content tool
        return (
            f"Mode switched to {normalized}. The current persona is {persona['display']}. "
            f"Please proceed by calling the relevant content tool (describe_current_concept, get_quiz_prompt, or get_teach_back_prompt) now."
        )

    def _require_concept(self, ctx: RunContext[Userdata]) -> TutorConcept:
        state = ctx.userdata.state
        try:
            return ctx.userdata.content.get(state.current_concept_id)
        except KeyError as exc:
            raise ToolError(str(exc)) from exc


def prewarm(proc: JobProcess):
    """Prewarm models and load tutor content."""
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["tutor_content"] = TutorContentLibrary.from_env()


async def entrypoint(ctx: JobContext):
    """Entry point for Day 4 active recall coach."""
    
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    content = ctx.proc.userdata["tutor_content"]
    state = TutorSessionState(current_concept_id=content.list_concepts()[0].id)
    userdata = Userdata(state=state, content=content)

    session = AgentSession[Userdata](
        userdata=userdata,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice=VOICE_PERSONAS["learn"]["voice"],
            style=VOICE_PERSONAS["learn"]["style"],
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    agent = TeachTheTutorAgent(userdata=userdata)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()
    logger.info("Day 4 Teach-the-Tutor agent is live and listening.")


if _name_ == "_main_":
    def prewarm_fnc(proc: JobProcess):
        from livekit.plugins import silero # Re-import here for local testing compatibility
        prewarm(proc) # Call the defined prewarm function

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm_fnc))