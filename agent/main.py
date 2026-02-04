import gradio as gr
from core.agent import AgentController
from core.user_manager import UserManager

def create_ui():
    agent = AgentController()
    user_mgr = UserManager()

    with gr.Blocks(title="PerTouch Agent") as demo:
        
        gr.HTML("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="color: #4A90E2; font-size: 2.5em; margin-bottom: 0;">PerTouch</h1>
                <p style="color: #666; font-style: italic;">Your Intelligent Visual Retouching Agent</p>
            </div>
        """)

        user_list_state = gr.State(user_mgr.get_all_users())

        with gr.Row():
            
            with gr.Column(scale=5):
                
                with gr.Row():
                    img_original = gr.Image(label="Original Image", type="numpy", height=600)
                    img_result = gr.Image(label="Current Result", type="numpy", height=600, interactive=False)
                
                gr.Markdown("### 👤 User Management")
                with gr.Row(variant="compact"):
                    user_dropdown = gr.Dropdown(
                        label="Select Role/User", 
                        choices=user_mgr.get_all_users(), 
                        value="Guest",
                        scale=2
                    )
                    new_user_name = gr.Textbox(
                        label="New Role Name", 
                        placeholder="Type name...", 
                        scale=2
                    )
                    create_user_btn = gr.Button(
                        "Create Role", 
                        variant="secondary",
                        scale=1
                    )
                
                user_status_msg = gr.Markdown("")

                gr.Markdown("### 🛠 System Logs")
                log_box = gr.TextArea(
                    label="Internal Thought Process", 
                    lines=10, 
                    max_lines=10,
                    interactive=False
                )

            # ================= RIGHT COLUMN (Interaction) =================
            with gr.Column(scale=5):
                gr.Markdown("### 💬 Agent Interaction")
                
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=900
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        show_label=False, 
                        placeholder="Type instruction (e.g., 'Global optimization')...",
                        scale=8
                    )
                    clean_btn = gr.Button("🧹 Clean", scale=1)
                    send_btn = gr.Button("🚀 Send", variant="primary", scale=1)
                
                gr.Examples(
                    examples=["Globally optimize this picture.", "Too dark, brighten it.", "Make the image color temperature cooler."],
                    inputs=msg_input
                )

        img_original.upload(
            fn=agent.init_session,
            inputs=[img_original, user_dropdown],
            outputs=[img_original, img_result, log_box, chatbot]
        )

        create_user_btn.click(
            fn=agent.handle_user_creation,
            inputs=[new_user_name],
            outputs=[user_dropdown, user_status_msg]
        )
        
        handle_chat_args = {
            "fn": agent.process_chat,
            "inputs": [msg_input],
            "outputs": [img_result, log_box, chatbot]
        }

        send_btn.click(**handle_chat_args).then(lambda: "", outputs=msg_input)
        msg_input.submit(**handle_chat_args).then(lambda: "", outputs=msg_input)

        clean_btn.click(
            fn=agent.reset_session,
            inputs=[],
            outputs=[img_original, img_result, log_box, chatbot]
        )

        demo.load(
            fn=agent.reset_session,
            inputs=[],
            outputs=[img_original, img_result, log_box, chatbot]
        )

    return demo

if __name__ == "__main__":
    app = create_ui()
    app.queue()
    app.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())