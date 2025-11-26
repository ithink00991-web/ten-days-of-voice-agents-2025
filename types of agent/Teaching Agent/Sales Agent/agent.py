import logging
import json
import os
from typing import Dict, Any, Optional, List 

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
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

LEAD_FILE_PATH = "captured_lead_data.json"

load_dotenv(".env.local")

COMPANY_NAME = "Tata Neu"
FAQ_CONTENT = {
    "what_it_does": "Tata Neu is a super-app that brings together the Tata Group's brand ecosystem—including shopping (e.g., BigBasket, Croma), travel (e.g., Air India), financial services, and payments—into a single platform.",
    "target_audience": "Our platform is primarily for consumers in India who want a unified loyalty and shopping experience across various retail, travel, and financial services under the trusted Tata brand.",
    "pricing_basics": "The Tata Neu app itself is free to download and use. Its value comes from earning and spending 'NeuCoins,' which are rewarded on purchases across all partner brands. Financial products offered through the app, like loans or credit cards, have their own specific pricing and fees.",
    "key_benefits": "The main benefit is the unified loyalty program (NeuPass) and seamless integration across all Tata brands, offering better rewards and a smoother checkout experience for users.",
    "free_tier": "There is no separate 'tier' since the app is free. The value is derived from user activity and rewards. We do offer promotional benefits that are effectively free perks."
}

LEAD_FIELDS = ["Name", "Company", "Email", "Role", "Use case", "Team size", "Timeline"]

class SDRSessionState:
    def _init_(self):
        self.lead_data: Dict[str, Any] = {field: None for field in LEAD_FIELDS}
        self.current_question: Optional[str] = None
        self.conversation_transcript: List[str] = []
        self.faq_hits: List[str] = []

    def get_missing_lead_fields(self) -> List[str]:
        return [field for field, value in self.lead_data.items() if value is None]


class SDRScriptAgent(Agent):
    def _init_(self, userdata: SDRSessionState) -> None:
        self.state = userdata
        
        instructions = f"""You are the Sales Development Representative (SDR) for {COMPANY_NAME}.
Your primary goal is to qualify the visitor, answer their questions based ONLY on the provided FAQ, and capture lead information.

*SDR Persona Rules:*
1.  *Greet Warmly:* Start by greeting the user and asking what brought them here.
2.  *Use FAQ:* If the user asks a product, pricing, or company question, use the answer_faq tool. *DO NOT invent details.*
3.  *Capture Lead Data:* Interweave lead questions naturally during the conversation using the capture_lead_data tool. Ask for missing fields one by one.
4.  *End Call:* When the user indicates they are done (e.g., "that's all," "thanks," "bye"), use the end_call_summary tool immediately to finish the session.

*Available FAQ Keys:* {', '.join(FAQ_CONTENT.keys())}
"""
        super()._init_(instructions=instructions)

    @function_tool
    async def answer_faq(self, context: RunContext, topic: str) -> str:
        """
        Use this tool to answer specific questions about {COMPANY_NAME} (features, pricing, audience).
        The 'topic' should be a keyword (e.g., 'pricing_basics', 'what_it_does', 'target_audience') that matches one of the available FAQ keys.
        
        Args:
            topic: The relevant FAQ key to retrieve the answer for.
        """
        topic = topic.lower().replace(' ', '').replace('-', '')
        
        for key, answer in FAQ_CONTENT.items():
            if topic in key or key in topic:
                context.userdata.faq_hits.append(key)
                return f"Regarding {COMPANY_NAME}, {answer}"

        return "I'm sorry, I don't have a pre-approved answer for that specific topic in my FAQ. Can I try to answer another question, or can I get your contact details?"

    @function_tool
    async def capture_lead_data(self, context: RunContext, field_name: Optional[str] = None, value: Optional[str] = None) -> str:
        """
        Use this tool to naturally ask for missing lead information or store a piece of information provided by the user.
        
        Args:
            field_name: The name of the field to update (e.g., 'Name', 'Email', 'Use case'). Optional if just checking missing fields.
            value: The value to store for the field. Optional if just checking missing fields.
        """
        if field_name and value:
            field_name = field_name.title()
            if field_name in LEAD_FIELDS:
                context.userdata.lead_data[field_name] = value
                logger.info(f"Captured lead data: {field_name}={value}")
        
        missing = context.userdata.get_missing_lead_fields()
        
        if not missing:
            return "Thank you! I believe I have all the key information I need to pass along to my team. How else can I help you today?"

        next_field = missing[0]
        context.userdata.current_question = next_field

        if next_field == "Name":
            return "That's helpful! Before we go further, can I just get your name?"
        elif next_field == "Email":
            return "And what would be the best email address to send our summary and follow-up resources to?"
        elif next_field == "Company":
            return "Great. What company are you currently with?"
        elif next_field == "Role":
            return "And what is your role or title at the company?"
        elif next_field == "Use case":
            return "What specific problem are you hoping to solve or what feature of Tata Neu interests you most?"
        elif next_field == "Team size":
            return "Roughly how many people are on the team that would be using our solution?"
        elif next_field == "Timeline":
            return "And finally, what's your timeline for implementing a new solution: immediately, within the next three months, or later this year?"
        
        return "I've updated the lead data. What is your next question?"

    @function_tool
    async def end_call_summary(self, context: RunContext) -> str:
        """
        Use this tool when the user signals the end of the conversation (e.g., "that's all," "thanks," "bye").
        It summarizes the lead data and saves it to a JSON file.
        """
        lead_data = context.userdata.lead_data
        
        # 1. Generate Verbal Summary for the User
        name = lead_data.get("Name", "the visitor")
        company = lead_data.get("Company", "an interested party")
        use_case = lead_data.get("Use case", "a potential integration")
        timeline = lead_data.get("Timeline", "an undefined timeline")

        summary_text = (
            f"Thank you, {name}, for your time today. To summarize: you are interested "
            f"in {COMPANY_NAME} for {use_case}. You are currently with {company}, and your "
            f"timeline is {timeline}. I will ensure a specialist follows up with you shortly."
        )

        # 2. Save Lead Data to JSON file
        try:
            # Load existing leads or start a new list
            if os.path.exists(LEAD_FILE_PATH):
                with open(LEAD_FILE_PATH, 'r') as f:
                    try:
                        leads = json.load(f)
                    except json.JSONDecodeError:
                        leads = []
            else:
                leads = []
            
            # Append new lead
            leads.append(lead_data)
            
            # Save back to file
            with open(LEAD_FILE_PATH, 'w') as f:
                json.dump(leads, f, indent=4)
            
            logger.info(f"Successfully saved lead data to {LEAD_FILE_PATH}")
        except Exception as e:
            logger.error(f"Failed to save lead data: {e}")
            summary_text += " (Note: There was an issue recording the data internally, but I have the information.)"

        return f"{summary_text} Thank you again and have a productive day!"


def prewarm(proc: JobProcess):
    """Prewarm models."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Set up session state
    session_state = SDRSessionState()
    
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline
    session = AgentSession(
        userdata=session_state,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
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

    # Start the session
    await session.start(
        agent=SDRScriptAgent(userdata=session_state),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if _name_ == "_main_":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))