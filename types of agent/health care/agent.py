import logging
import json
from datetime import datetime
import os # NEW: Added for file management

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool, # UNCOMMENTED
    RunContext     # UNCOMMENTED
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# NEW: Define the JSON log file path
WELLNESS_LOG_FILE = "wellness_log.json"


class Assistant(Agent):
    def __init__(self) -> None:
        # NEW: Updated instructions for the Health & Wellness Companion
        super().__init__(
            instructions="""You are Luna, a supportive, realistic, and grounded Health & Wellness Voice Companion.
            Your role is to conduct a short daily check-in with the user about their mood, energy, and daily objectives.
            You must avoid diagnosis or medical claims.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            
            Before starting the conversation, use the `get_past_checkin_summary` tool to retrieve any historical context.
            When the check-in is complete (mood, energy, and 1-3 objectives are gathered), you MUST use the `save_checkin_data` tool to persist the session details, and then recap the session and confirm with the user before concluding.
            """,
        )

    def _read_wellness_log(self):
        """Helper function to read all previous check-ins."""
        if not os.path.exists(WELLNESS_LOG_FILE):
            return []
        try:
            with open(WELLNESS_LOG_FILE, 'r') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            logger.warning(f"Could not read or parse {WELLNESS_LOG_FILE}. Starting fresh.")
            return []

    # NEW TOOL 1: Save data to JSON
    @function_tool
    async def save_checkin_data(self, context: RunContext, mood_summary: str, objectives: list[str]):
        """Use this tool to persist the key data from the current check-in session. 
        It should be called only once the agent has gathered the user's mood, energy, and 1-3 intentions/objectives.

        Args:
            mood_summary: A short, simple summary of the user's self-reported mood and energy (e.g., 'Feeling a bit stressed, but energy is high.').
            objectives: A list of 1 to 3 simple, practical goals or intentions for the day.
        """

        log_data = self._read_wellness_log()
        new_entry = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mood_summary": mood_summary,
            "objectives": objectives,
            "agent_summary_sentence": f"Check-in logged successfully on {datetime.now().strftime('%Y-%m-%d')}."
        }
        log_data.append(new_entry)

        with open(WELLNESS_LOG_FILE, 'w') as f:
            json.dump(log_data, f, indent=4)

        logger.info(f"Saved wellness check-in: {new_entry}")
        
        # This message is what the LLM will see, informing it the save was successful.
        return "Check-in data has been successfully saved to the wellness log."

    # NEW TOOL 2: Get a summary of the most recent check-in for context
    @function_tool
    async def get_past_checkin_summary(self, context: RunContext):
        """Use this tool to retrieve a summary of the most recent check-in to provide historical context to the user.
        This should be called at the very start of a new session to personalize the greeting.
        """
        log_data = self._read_wellness_log()
        if log_data:
            latest = log_data[-1]
            date_str = latest['date'].split(' ')[0]
            summary = latest.get('mood_summary', 'no mood recorded')
            obj_count = len(latest.get('objectives', []))
            
            # The LLM uses this information to generate its first response
            return f"Your last check-in was on {date_str}. You reported: '{summary}'. You had {obj_count} objectives set. Use this to ask a relevant opening question."
        
        return "No past wellness check-in data found."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-IN-Anisha", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))