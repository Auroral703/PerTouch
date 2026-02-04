import json
import base64
import os
import cv2
import numpy as np
from config import Config
from openai import OpenAI
from core.prompts import SYSTEM_PROMPT

class VLMInterface:
    def __init__(self):
        self.client = OpenAI(
            api_key=Config.API_KEY,
            base_url=Config.API_URL
        )
        self.model_name = Config.VLM_MODEL_NAME

    def _encode_image(self, image_np):
        """Convert image_np to base64"""
        if image_np is None: 
            return None
        _, buffer = cv2.imencode(".jpg", image_np)
        return base64.b64encode(buffer).decode("utf-8")

    def _call_vlm(self, prompt, images=[]):
        """
        Generic VLM caller.
        images: list of numpy arrays [image1, image2...]
        """
        content = []
        
        # Add Images
        for img in images:
            if img is not None:
                b64_str = self._encode_image(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
                })
        
        # Add Text Prompt
        content.append({"type": "text", "text": prompt})

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content}
                ],
                stream=False,
                extra_body={'enable_thinking': False} 
            )
            
            answer = completion.choices[0].message.content
            
            # Clean up potential markdown formatting
            clean_answer = answer.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_answer)
            
        except Exception as e:
            # Fallback JSON to prevent crash
            return {"think": f"Error parsing response: {str(e)}", "error": True}


    def get_params_from_memory(self, history, image):
        """
        Analyzes scene tags and recalls user preferences.
        """
        history_str = json.dumps(history, ensure_ascii=False) if history else "[]"
        
        prompt = f"""
        **Task**: Analyze the input image scene and suggest initial retouching parameters based on user history.
        
        **User Preference History**:
        {history_str}
        
        **Steps**:
        1. Identify scene tags (max 8, e.g., 'sunset', 'portrait', 'indoor').
        2. Check if the current scene matches any record in the User History.
        3. If a match is found, retrieve the 'set_state' parameters. 
        4. If no match, generate a conservative 'default' parameter set or leave empty.
        
        **Response Format (JSON)**:
        {{
            "think": "Describe the scene analysis and memory matching process...",
            "scene_tags": ["tag1", "tag2"],
            "set_state": {{
                "global": {{"brightness": 0.1}}, 
                "sky": {{"colorfulness": 0.2}} 
            }},
            "suggestion": "Brief greeting to user explaining what you applied."
        }}
        """
        res = self._call_vlm(prompt, images=[image])
        return res.get("think", ""), res.get("set_state", {}), res.get("scene_tags", []), res.get("suggestion", "")

    def classify_instruction(self, instruction):
        """
        Distinguishes between Weak (Global/Style) and Strong (Regional/Specific).
        """
        prompt = f"""
        **Task**: Classify the user instruction into one of three types.
        
        **Instruction**: "{instruction}"
        
        **Definitions**:
        1. "weak_optimize": Vague requests like "make it look better", "optimize image". No specific attributes mentioned.
        2. "weak_style": Requests for artistic styles like "Cyberpunk", "Oil Painting", "Kodak Film".
        3. "strong_analysis": Specific requests mentioning Regions (sky, grass, face) OR Attributes (brightness, contrast, warm, cold).
        
        **Response Format (JSON)**:
        {{
            "think": "Analyze keywords in the instruction...",
            "instr_type": "weak_optimize" | "weak_style" | "strong_analysis"
        }}
        """
        res = self._call_vlm(prompt)
        return res.get("think", ""), res.get("instr_type", "strong_analysis")

    def weak_style_analysis(self, instruction, current_image):
        """
        Handles style transfer logic by mapping abstract styles to concrete parameters.
        """
        prompt = f"""
        **Task**: Translate the user's artistic style requirement into concrete image parameters.
        
        **Instruction**: "{instruction}"
        
        **Steps**:
        1. Analyze what physical properties define this style (e.g., 'Cyberpunk' -> high contrast, cool temp, high saturation).
        2. Generate a 'update_state' map. 
        3. You can use 'global' or specific regions if you detect them in the image.
        
        **Response Format (JSON)**:
        {{
            "think": "Deconstruct the style into parameters...",
            "update_state": {{
                "global": {{"contrast": 0.2, "temperature": -0.3}},
                "neon_lights": {{"colorfulness": 0.5}}
            }},
            "simple_state": "I am applying a [Style Name] look..."
        }}
        """
        res = self._call_vlm(prompt, images=[current_image])
        return res.get("think", ""), res.get("update_state", {}), res.get("simple_state", "")

    def strong_analysis(self, instruction, pre_image, current_image, last_action=None):
        """
        Refers to `retouch_process` in chains.py.
        The "Rethink" Logic: Compares Pre vs Current to validate user satisfaction.
        """
        
        # Mode 1: First Iteration (No pre_image)
        if pre_image is None:
            # ... (This part remains the same) ...
            prompt = f"""
            **Task**: Parse the Strong Instruction to identify the Target Region and Attributes.
            
            **Instruction**: "{instruction}"
            
            **Steps**:
            1. Identify the Target Region (e.g., 'sky', 'food'). If not specified, use 'global'.
            2. Identify the Target Attribute (brightness, contrast, colorfulness, temperature).
            3. Estimate an initial adjustment value (-1.0 to 1.0) based on the intensity of the user's words (e.g., "slightly" -> 0.1, "very" -> 0.4).
            
            **Response Format (JSON)**:
            {{
                "think": "Reasoning about region extraction and intensity...",
                "satisfied": false,
                "update_state": {{ "region_name": {{ "attribute_name": 0.3 }} }},
                "simple_state": "Adjusting [region]..."
            }}
            """
            images_to_send = [current_image]

        # Mode 2: Rethink / Evaluation (Comparing Pre vs Current)
        else:
            # --- MODIFICATION HERE: Add last_action to the prompt ---
            action_str = json.dumps(last_action) if last_action else "None"
            
            prompt = f"""
            **Task**: Evaluate if the change from Image 1 (Previous) to Image 2 (Current) satisfies the user instruction.
            
            **Instruction**: "{instruction}"
            
            **Last Action Taken**: {action_str}
            
            **Steps**:
            1. Compare Image 1 and Image 2 visually. Did the requested attribute change?
            2. Evaluate the effect of the **Last Action**. 
               - If the Last Action was {action_str}, did it have the desired visual impact?
            3. Does the MAGNITUDE of change match the user's intent?
               - If user said "make it MUCH brighter" but it only changed slightly -> Not Satisfied (Need to increase value).
               - If user said "a little bit" but it changed too much -> Not Satisfied (Need to reverse/decrease value).
            4. If Satisfied, return "satisfied": true.
            5. If Not Satisfied, propose a NEW adjustment value (delta) to fix it.
            
            **Response Format (JSON)**:
            {{
                "think": "Compare Image 1 vs Image 2. The previous action {action_str} resulted in... The user wanted... Therefore I need to...",
                "satisfied": true | false,
                "update_state": {{ "region_name": {{ "attribute_name": 0.1 }} }} (Only if not satisfied),
                "simple_state": "The previous adjustment was too weak, increasing it..."
            }}
            """
            images_to_send = [pre_image, current_image]

        res = self._call_vlm(prompt, images=images_to_send)
        
        if "error" in res:
            return "Error in VLM", True, {}, "Error processing request."
            
        return res.get("think", ""), res.get("satisfied", False), res.get("update_state", {}), res.get("simple_state", "")