import logging
import json 
import os 

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
    function_tool, 
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are 'Shadow Brews Barista', a friendly and welcoming barista for a premium coffee shop. Your goal is to take a customer's coffee order and fill out all fields of the available tool's order state.

            Your responses must be friendly, encouraging, and focused only on gathering the following order details: drinkType (e.g., Latte, Cappuccino, Espresso, Tea), size (e.g., Small, Large), milk (e.g., Oat, Whole), any extras (e.g., Whipped Cream, Extra Shot), and the customer's name.

            If a user provides an incomplete order, you must ask a polite, clarifying question to get the next missing piece of information.

            Once you have all the information required by the 'place_coffee_order' tool, call the tool immediately to finalize the order. Do not ask for confirmation before calling the tool if all information is present.
            """,
        )

    @function_tool 
    async def place_coffee_order(
        self,
        context: RunContext,
        drinkType: str,
        size: str,
        milk: str,
        extras: list[str],
        name: str,
    ):
        """Use this tool to finalize a customer's coffee order and save the details once all fields are known.
        
        Args:
            drinkType: The main type of coffee or beverage (e.g., Latte, Cappuccino, Espresso, Tea).
            size: The desired size of the drink (e.g., Small, Medium, Large).
            milk: The type of milk to be used (e.g., Whole, Skim, Oat, Almond).
            extras: A list of any additional ingredients (e.g., Whipped Cream, Extra Shot, Vanilla Syrup, Caramel). Use an empty list if none.
            name: The customer's name for the order.
        """

        order = {
            "drinkType": drinkType,
            "size": size,
            "milk": milk,
            "extras": extras,
            "name": name,
        }

        # REQUIRED STEP: Save the order to a JSON file
        file_path = "day2_order_summary.json"
        
        # Get the path to the backend directory, or current directory as a fallback
        output_dir = os.path.join(os.getcwd(), "backend") 
        if not os.path.exists(output_dir):
            output_dir = os.getcwd() 

        final_path = os.path.join(output_dir, file_path)

        with open(final_path, 'w') as f:
            json.dump(order, f, indent=4)
        
        # Return a friendly confirmation message to the LLM for it to speak back to the user
        return f"Order for {name} has been placed. Details saved to {final_path}. The customer ordered a {size} {drinkType} with {milk} and the following extras: {', '.join(extras) if extras else 'None'}."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using Murf Falcon, Gemini, Deepgram, and LiveKit
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
                voice="en-US-matthew", 
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
