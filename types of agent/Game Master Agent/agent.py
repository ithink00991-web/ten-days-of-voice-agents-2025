import logging
import os
import sys
import asyncio
from typing import Annotated

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    function_tool,
    RunContext,
    tokenize
)
from livekit.plugins import google, deepgram, silero, murf
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Import tools
try:
    from src.gamemaster_tools import WorldState
except ImportError:
    from gamemaster_tools import WorldState

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), "../.env.local")
load_dotenv(env_path)

if not os.getenv("GOOGLE_API_KEY"):
    load_dotenv() # Fallback

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is not set.")

# Configure Logging
logger = logging.getLogger("gamemaster")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("gamemaster_debug.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

class GameMasterAgent(Agent):
    def __init__(self):
        self.world = WorldState()
        super().__init__(
            instructions=self._get_instructions()
        )

    def _get_instructions(self):
        state = self.world.get_state()
        return (
            "You are the Game Master \"Benimaru\" for a fantasy tabletop RPG. "
            "Your goal is to guide the player through an immersive adventure. "
            "1. Describe the current scene vividly using the provided world state. "
            "2. React to the player's actions. If an action is risky, use the 'roll_dice' tool. "
            "3. Manage inventory using 'add_item' and 'remove_item' tools. "
            "4. Keep track of the player's health and status. "
            "5. Always end your turn by asking 'What do you do?' or prompting for action. "
            "6. Maintain a consistent tone: Mysterious, adventurous, but fair. "
            f"Current World State: {state}"
        )

    @function_tool
    async def roll_dice(self, context: RunContext, sides: int = 20, reason: str = "Action check") -> str:
        """
        Roll a die to determine the outcome of a risky action.
        Args:
            sides: Number of sides on the die (default 20).
            reason: The reason for the roll (e.g., 'Attack goblin', 'Climb wall').
        """
        result = self.world.roll_dice(sides)
        outcome = "Success" if result >= 10 else "Failure" # Simple threshold
        return f"Rolled d{sides} for {reason}: {result}. Outcome: {outcome}"

    @function_tool
    async def check_inventory(self, context: RunContext) -> str:
        """Check the player's current inventory."""
        return self.world.get_inventory_description()

    @function_tool
    async def add_item(self, context: RunContext, item_name: str) -> str:
        """Add an item to the player's inventory."""
        return self.world.add_inventory_item(item_name)

    @function_tool
    async def remove_item(self, context: RunContext, item_name: str) -> str:
        """Remove an item from the player's inventory."""
        return self.world.remove_inventory_item(item_name)

    @function_tool
    async def get_character_sheet(self, context: RunContext) -> str:
        """Get the full character sheet including HP and stats."""
        return self.world.get_character_sheet()

    @function_tool
    async def save_game(self, context: RunContext) -> str:
        """Save the current game progress."""
        return self.world.save_game()

    @function_tool
    async def load_game(self, context: RunContext) -> str:
        """Load a previously saved game."""
        return self.world.load_game()

def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception as e:
        logger.error(f"Error in prewarm: {e}", exc_info=True)

async def entrypoint(ctx: JobContext):
    try:
        logger.info(f"Connecting to room {ctx.room.name}")
        
        try:
            turn_model = MultilingualModel()
        except Exception as e:
            logger.warning(f"Turn detector init failed: {e}")
            turn_model = None

        session = AgentSession(
            stt=deepgram.STT(model="nova-3"),
            llm=google.LLM(
                model="gemini-2.5-flash",
                api_key=os.getenv("GOOGLE_API_KEY")
            ),
            tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
            vad=ctx.proc.userdata["vad"],
            turn_detection=turn_model,
        )
        
        agent = GameMasterAgent()
        await session.start(agent=agent, room=ctx.room)
        
        await ctx.connect()
        
        # Initial greeting and scene description
        initial_state = agent.world.get_state()
        intro = (
            f"Welcome, traveler. {initial_state['location']['description']} "
            "You have a map, a dagger, and some water. What is your first move?"
        )
        await session.say(intro, allow_interruptions=True)
        
    except Exception as e:
        logger.error(f"Error in entrypoint: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint,
                prewarm_fnc=prewarm,
            ),
        )
    except Exception as e:
        logger.critical(f"Critical error in main: {e}", exc_info=True)