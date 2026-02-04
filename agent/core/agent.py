import time
import json
from core.memory import Memory
from core.prompts import SYSTEM_PROMPT
from core.user_manager import UserManager
from interfaces.vlm_interface import VLMInterface
from interfaces.param_map_interface import ParamMapInterface
from interfaces.retouch_interface import RetouchInterface
from config import Config

class AgentController:
    def __init__(self):
        self.vlm = VLMInterface()
        self.param_interface = ParamMapInterface()
        self.retouch_interface = RetouchInterface()
        self.user_manager = UserManager()
        
        # Session State
        self.current_user = None
        self.memory = None
        self.original_image = None # Numpy array
        self.current_image = None  # Numpy array
        self.pre_image = None      # Numpy array
        self.scene_tags = []
        self.history = []  # Chat history for UI
        self.logs = ""

    def _add_log(self, tag, message):
        """Record a log message with a timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        self.logs += f"[{timestamp}] [{tag}] {message}\n"
    
    def init_session(self, image, user_id):
        """Initialize session: triggered when an image is uploaded"""
        self.logs = "" # Reset logs
        self._add_log("System", f"Initialising session for user: {user_id}")

        if image is None:
            return None, None, "Please upload an image.", []
        
        self.original_image = image
        self.current_user = user_id
        self.memory = Memory(user_id)
        
        # Reset State
        self.param_interface.reset()

        self.param_interface.set_image(self.original_image)
        
        think, set_state, scene_tags, suggestion = self.vlm.get_params_from_memory(self.history, self.original_image)
        # Based on historical scenes, determine if there are any借鉴able elements in the current image and provide suggestions.
        # If not, param_interface.state remains empty; if so, it should be populated based on historical records.
        # Input: history, original_image
        # Output: think, set_state, scene_tags, suggestion
        
        self._add_log("VLM", think)
        self._add_log("VLM", f"Scene Analysis: {', '.join(scene_tags)}")
        print(set_state)
        self.param_interface.set_state(set_state)
        
        self.scene_tags = scene_tags
        self.history.append({"role": "assistant", "content": suggestion})
        self._add_log("System", "Initial render complete.")

        return self.original_image, self.current_image, self.logs, self.history

    def reset_session(self):
        """Reset the session state"""
        # 1. Clear original, current, and pre-image
        self.original_image = None
        self.current_image = None
        self.pre_image = None
        
        # 2. Reset chat history and logs
        self.history = []
        self.logs = ""
        self.scene_tags = []
        
        # 3. Reset parameter interface state
        self.param_interface.reset()
        
        # 4. Log the reset action
        self._add_log("System", "Session has been reset.")
        
        # 5. Return empty values to the UI components:
        return None, None, "", []

    def handle_user_creation(self, new_name):
        success = self.user_manager.create_user(new_name)
        if success:
            return self.user_manager.get_all_users(), f"User '{new_name}' created!"
        else:
            return self.user_manager.get_all_users(), f"User '{new_name}' already exists or invalid."

    def gen_current_image(self):
        param_map = self.param_interface.generate_parameter_map()
        self.current_image = self.retouch_interface.process(self.original_image, param_map)

    def process_chat(self, message):
        if self.original_image is None:
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": "Please upload an image first."})
            yield self.current_image, "Error: No image loaded.", self.history
            return

        self.history.append({"role": "user", "content": message})
        yield self.current_image, self.logs, self.history

        # --- Step 1: Instruction Classification and Initial Analysis ---
        # We need VLM to return the instruction type: 'weak_optimize', 'weak_style', 'strong_global', 'strong_regional'
        think, instr_type = self.vlm.classify_instruction(message)
        # Analyze instruction type and enter different workflows
        # Input: Instruction
        # Output: think, instr_type
        self._add_log("VLM", think)
        self._add_log("System", f"Workflow selected: {instr_type}")

        # --- Step 2: Enter Different Branches Based on Type ---
        if instr_type.startswith("weak"):
            # 【Weak Instruction Branch】: Execute only once

            if instr_type == "weak_optimize":
                self._add_log("Action", "Executing one-click optimization.")

                self.gen_current_image()
                
                self.history.append({"role": "assistant", "content": "Finish weak optimization."})
                yield self.current_image, self.logs, self.history
            
            else:
                if self.current_image is None:
                    self.gen_current_image()
                
                think, update_state, simple_state = self.vlm.weak_style_analysis(message, self.current_image)
                self._add_log("VLM", think)
                self._add_log("Action", f"Update: {json.dumps(update_state)}")
                # Decompose user style instruction intent and try to break it down into a series of atomic property actions
                # Input: Instruction, current_image
                # Output: think, update_state, simple_state

                self.param_interface.update_state(update_state)

                self.gen_current_image()
                
                self.history.append({"role": "assistant", "content": simple_state})
                yield self.current_image, self.logs, self.history

        else:
            # 【Strong Instruction Branch】: Trigger Rethink mechanism
            
            attempts = 0
            satisfied = False
            self.pre_image = None
            last_action = {} # Initialize variable to store the previous action
            
            self._add_log("System", "Entering Rethink loop...")

            if self.current_image is None:
                self.gen_current_image()

            # First Analysis (No pre_image, No last_action)
            think, satisfied, update_state, simple_state = self.vlm.strong_analysis(
                message, 
                self.pre_image, 
                self.current_image,
                last_action=None 
            )

            while not satisfied and attempts < Config.MAX_RETHINK_RETRIES:
                attempts += 1
                self._add_log("Loop", f"Attempt #{attempts}")
                self._add_log("VLM", think)
                self._add_log("Action", f"Refining parameters: {json.dumps(update_state)}")
                
                # Apply the update
                self.param_interface.update_state(update_state)
                
                # Store this action to pass to the next analysis
                last_action = update_state 
                
                # Render
                self.pre_image = self.current_image
                self.gen_current_image()
                
                self.history.append({"role": "assistant", "content": simple_state})
                yield self.current_image, self.logs, self.history
                
                # Rethink: Now we pass pre_image, current_image, AND last_action
                think, satisfied, update_state, simple_state = self.vlm.strong_analysis(
                    message, 
                    self.pre_image, 
                    self.current_image,
                    last_action=last_action # <--- Pass the context here
                )
                
                if satisfied:
                    self._add_log("System", "Agent is satisfied with the result.")
                    self.history.append({"role": "assistant", "content": "I think this looks good now!"})
                    yield self.current_image, self.logs, self.history