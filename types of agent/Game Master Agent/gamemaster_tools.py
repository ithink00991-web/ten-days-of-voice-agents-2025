import json
import random
import logging
import os

logger = logging.getLogger("gamemaster-tools")

class WorldState:
    def __init__(self):
        self.state = {
            "character": {
                "name": "dark",
                "class": "Adventurer",
                "hp": 20,
                "max_hp": 20,
                "inventory": ["gold coin", "Rusty sword", "Water bottle"],
                "status": "Healthy"
            },
            "location": {
                "name": "The Crossroads",
                "description": "You stand at a dusty crossroads. To the north lies the Dark Forest, to the east the Village of Oakhaven.",
                "known_paths": ["North", "East", "South"]
            },
            "events": [],
            "quests": [
                {"name": "Find the Lost Relic", "status": "Active", "description": "Rumors say a powerful relic is hidden in the Dark Forest."}
            ]
        }

    def get_state(self) -> dict:
        """Returns the current world state."""
        return self.state

    def update_location(self, name: str, description: str, paths: list[str]):
        """Updates the current location."""
        self.state["location"] = {
            "name": name,
            "description": description,
            "known_paths": paths
        }
        logger.info(f"Location updated to: {name}")

    def update_character_status(self, hp: int = None, status: str = None):
        """Updates character HP and status."""
        if hp is not None:
            self.state["character"]["hp"] = hp
        if status is not None:
            self.state["character"]["status"] = status
        logger.info(f"Character status updated: HP={self.state['character']['hp']}, Status={self.state['character']['status']}")

    def add_inventory_item(self, item: str):
        """Adds an item to the inventory."""
        if item not in self.state["character"]["inventory"]:
            self.state["character"]["inventory"].append(item)
            logger.info(f"Added item: {item}")
            return f"Added {item} to inventory."
        return f"{item} is already in inventory."

    def remove_inventory_item(self, item: str):
        """Removes an item from the inventory."""
        if item in self.state["character"]["inventory"]:
            self.state["character"]["inventory"].remove(item)
            logger.info(f"Removed item: {item}")
            return f"Removed {item} from inventory."
        return f"{item} not found in inventory."

    def log_event(self, event: str):
        """Logs a significant event."""
        self.state["events"].append(event)
        logger.info(f"Event logged: {event}")

    def roll_dice(self, sides: int = 20) -> int:
        """Rolls a die with the specified number of sides."""
        result = random.randint(1, sides)
        logger.info(f"Rolled d{sides}: {result}")
        return result

    def get_inventory_description(self) -> str:
        """Returns a formatted string of the inventory."""
        items = ", ".join(self.state["character"]["inventory"])
        return f"Inventory: {items}"

    def get_character_sheet(self) -> str:
        """Returns a formatted character sheet."""
        char = self.state["character"]
        return (f"Name: {char['name']} ({char['class']})\n"
                f"HP: {char['hp']}/{char['max_hp']} ({char['status']})\n"
                f"Inventory: {', '.join(char['inventory'])}")

    def save_game(self, filename: str = "gamestate.json") -> str:
        """Saves the current world state to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.state, f, indent=2)
            logger.info(f"Game saved to {filename}")
            return f"Game successfully saved to {filename}."
        except Exception as e:
            logger.error(f"Error saving game: {e}")
            return "Failed to save game."

    def load_game(self, filename: str = "gamestate.json") -> str:
        """Loads the world state from a JSON file."""
        try:
            if not os.path.exists(filename):
                return "No saved game found."
            
            with open(filename, 'r') as f:
                self.state = json.load(f)
            logger.info(f"Game loaded from {filename}")
            return "Game loaded successfully. Welcome back, traveler."
        except Exception as e:
            logger.error(f"Error loading game: {e}")
            return "Failed to load game."