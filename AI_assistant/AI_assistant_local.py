from http import server
import gradio as gr
import json
import os
import sys
import subprocess
import datetime
from pathlib import Path
import torch
import requests
import pandas as pd
import zipfile
import tempfile
import shutil

# Load system prompt from proto files
current_dir = os.path.dirname(os.path.abspath(__file__))

def load_prompt(typ):
    """Load prompt content from proto files"""
    with open(os.path.join(current_dir, 'proto', f'{typ}.proto'), 'r') as f:
        return f.read()

def is_valid_python_path(python_path: str) -> bool:
    """Validate if a given path points to a valid Python executable"""
    try:
        path = Path(python_path).resolve()
        if not path.exists() or not path.is_file():
            return False
        if os.name == 'posix' and not os.access(path, os.X_OK):
            return False
        result = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=2)
        return result.returncode == 0 and "Python" in result.stdout
    except Exception:
        return False

# Initialize global variables
ai_types = ['sort', 'professor', 'LVK', 'LTT', 'PTA', 'HE', 'code']
type_token = {'sort': 10, 'professor': 200, 'LVK': 400, 'LTT': 400, 'PTA': 400, 'HE': 400, 'code': 500}
ai_dir = {}  # Stores AI models for different types

def test_api_key(provider, model_name, api_key):
    """Validate API key for different AI providers"""
    try:
        if provider == "openai":
            url = "https://api.openai.com/v1/models"
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                model_ids = [m["id"] for m in response.json()["data"]]
                if model_name in model_ids:
                    return True, ''
                else:
                    return False, f"[OpenAI] API key valid, but model '{model_name}' not found."
            else:
                return False, f"[OpenAI] API key invalid or unauthorized: {response.status_code}"

        elif provider == "deepseek":
            url = "https://api.deepseek.com/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            response = requests.options(url, headers=headers, timeout=10)
            if response.status_code in [200, 204]:
                return True, "[DeepSeek] Connection successful. Note: API key validity cannot be pre-verified for DeepSeek."
            else:
                return False, f"[DeepSeek] Connection failed: {response.status_code}"
    except Exception as e:
        return False, f"[Error] Exception during API validation: {e}"

def set_ai(model_mode, local_model_name, use_gpu_flag, remote_model_name, deepseek_model, openai_model, other_model, api_key, python_path_1, python_path_2):
    """Initialize AI models based on user configuration"""
    warning = ''

    # Handle Python path
    python_path_1 = sys.executable if python_path_1 == 'python' else python_path_1
    python_path_2 = sys.executable if python_path_2 == 'python' else python_path_2
    if not is_valid_python_path(python_path_1):
        return "Invalid python path!", False, gr.update(visible=True)
    
    if not is_valid_python_path(python_path_2):
        return "Invalid python path!", False, gr.update(visible=True)
    
    # Initialize AI models
    global ai_dir
    if model_mode == "local":
        warning += 'Local models are still experimental and may not work properly\n'
        if use_gpu_flag:
            if not torch.cuda.is_available():
                use_gpu_flag = False
                warning += 'GPU not available, using CPU\n'
        from local_llm import LocalLLM as LLM
        for typ in ai_types:
            ai_dir[typ] = LLM(
                model_type=local_model_name,
                system_prompt=load_prompt(typ),
                max_new_tokens=type_token[typ],
                api_key=None,
                use_gpu=use_gpu_flag
            )
    else:
        from remote_llm import RemoteLLM as LLM
        if remote_model_name == "deepseek":
            model = deepseek_model
        elif remote_model_name == "openai":
            model = other_model if openai_model == "other model" else openai_model
        
        if not api_key:
            return "Empty API key!", False, gr.update(visible=True)
            
        state, info = test_api_key(remote_model_name, model, api_key)
        if not state:
            return info, False, gr.update(visible=True)
        else:
            warning += info
            for typ in ai_types:
                ai_dir[typ] = LLM(
                    model_type=remote_model_name,
                    model_name=model,
                    system_prompt=load_prompt(typ),
                    api_key=api_key
                )
    return warning + "Model setup successful!", True, gr.update(visible=False)

def create_new_dialogue(savepath, user_input):
    """Create a new dialogue directory structure"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dialogue_name = timestamp + user_input[:20].replace(" ", "_")
    global dialogue_path
    dialogue_path = os.path.join(savepath, dialogue_name)
    os.makedirs(dialogue_path, exist_ok=True)
    os.chmod(dialogue_path, 0o777)
    
    # Create subdirectories
    for sub in ['pyfiles', 'figures', 'dialogues', 'catalogs']:
        os.makedirs(os.path.join(dialogue_path, sub), exist_ok=True)
        os.chmod(os.path.join(dialogue_path, sub), 0o777)
    
    # Initialize empty JSON files
    for file in ['sort.json', 'message.json', 'code.json']:
        with open(os.path.join(dialogue_path, 'dialogues', file), "w", encoding="utf-8") as f:
            json.dump([], f, indent=4, ensure_ascii=False)
        os.chmod(os.path.join(dialogue_path, 'dialogues', file), 0o777)


def save_ai_code_to_file(ai_response: str, save_path: str, add_prefix: bool = False):
    """Extract and save code from AI response"""
    lines = ai_response.strip().split('\n')
    if lines and lines[0].strip().startswith('```'):
        lines = lines[1:]
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    code_str = '\n'.join(lines)
    if add_prefix:
        prefix = (
            "import xspec\n"
            "xspec.Xset.chatter = 0\n"
            "xspec.Xset.show = 0\n\n"
        )
        code_str = prefix + code_str
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(code_str)


def gradio_run(message, history, settings_available, savepath, python_path_1, python_path_2):
    """Main chat processing function"""
    if not settings_available:
        yield "Please configure the AI model first"
        return
    
    # Initialize or load conversation history
    global dialogue_path, timestamp
    if not history:
        dialogue = []
        code = []
        sorts = []
        create_new_dialogue(savepath, message)
    else:
        dialogue = history
        with open(os.path.join(dialogue_path, 'dialogues', 'code.json'), "r", encoding="utf-8") as f:
            code = json.load(f)
        with open(os.path.join(dialogue_path, 'dialogues', 'sort.json'), "r", encoding="utf-8") as f:
            sorts = json.load(f)

    # Add user message to history
    dialogue.append({"role": "user", "content": message})
    sorts.append({"role": "user", "content": message})
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine response type using 'sort' AI
    response_type = None
    for _ in range(5):
        response_type = ai_dir['sort'](sorts, stream=False)
        if response_type in ['professor', 'LVK', 'LTT', 'PTA', 'HE']:
            break
    else:
        yield "Please try rephrasing your question!"
        return
    
    # Save sorted response
    sorts.append({"role": "assistant", "content": response_type})
    with open(os.path.join(dialogue_path, 'dialogues', 'sort.json'), "w", encoding="utf-8") as f:
        json.dump(sorts, f, indent=4, ensure_ascii=False)

    # Generate and stream response
    reply = ""
    for part in ai_dir[response_type](dialogue, stream=True):
        reply += part
        yield reply
    full_response = ''.join(reply)
    
    # Save final response
    dialogue.append({"role": "assistant", "content": full_response})
    with open(os.path.join(dialogue_path, 'dialogues', 'message.json'), "w", encoding="utf-8") as f:
        json.dump(dialogue, f, indent=4, ensure_ascii=False)

    # Handle code execution for specific response types
    if response_type in ['LVK', 'LTT', 'PTA', 'HE']:
        full_response += "\nExecuting code..."
        yield full_response
        
        # Generate and save code
        code.append({
            "role": "user", 
            "content": f"Q: {message}\nA: {full_response}\nPath: {dialogue_path}\nMsg ID: {timestamp}"
        })
        ai_code = ai_dir['code'](code, stream=False)
        code.append({"role": "assistant", "content": ai_code})
        with open(os.path.join(dialogue_path, 'dialogues', 'code.json'), "w", encoding="utf-8") as f:
            json.dump(code, f, indent=4, ensure_ascii=False)

        # Execute code
        py_file = os.path.join(dialogue_path, 'pyfiles', f'{timestamp}.py')
        if response_type == 'HE':
            save_ai_code_to_file(ai_code, py_file, add_prefix=True)
            result = subprocess.run(["conda", "run", "-n", 'hetoolbox', python_path_2, py_file], capture_output=True, text=True)
        else:
            save_ai_code_to_file(ai_code, py_file)
            result = subprocess.run([python_path_1, py_file], capture_output=True, text=True)
        
        # Handle execution results
        if result.returncode:
            print(result.stderr)
            full_response += f"\nExecution failed:\n{result.stderr}"
        else:
            full_response += "\nExecution result:\n" + repr(result.stdout)
        yield full_response
    
def clear_ui():
    """Reset UI elements"""
    return None, None, []
    
def update_ui(code_display, catalog_display, image_display):
    """Update UI with generated content"""
    # Get file lists
    py_dir = os.path.join(dialogue_path, 'pyfiles')
    img_dir = os.path.join(dialogue_path, 'figures')
    cat_dir = os.path.join(dialogue_path, 'catalogs')
    
    py_files = os.listdir(py_dir)
    images = os.listdir(img_dir)
    catalogs = os.listdir(cat_dir)
    
    # Find matching files based on timestamp
    matched_py = next((f for f in py_files if timestamp in f), None)
    matched_images = [f for f in images if timestamp in f]
    matched_catalog = next((f for f in catalogs if timestamp in f), None)

    # Update code display
    new_code = code_display
    if matched_py:
        with open(os.path.join(py_dir, matched_py), 'r', encoding='utf-8') as f:
            new_code = f.read()

    # Update image display
    new_images = [os.path.join(img_dir, img) for img in matched_images]
    if image_display:
        new_images.extend(image_display)

    # Update catalog display
    new_catalog = catalog_display
    if matched_catalog:
        with open(os.path.join(cat_dir, matched_catalog), "r", encoding="utf-8") as f:
            try:
                new_catalog = pd.DataFrame(json.load(f))
            except Exception as e:
                new_catalog = None

    return (
        gr.Code(language="python", value=new_code) if new_code else gr.Code(language="python", value=code_display),
        new_catalog,
        new_images
    )

def read_dialogue(zip_file_path, savepath):
    """Import dialogue from zip archive"""
    base_name = os.path.splitext(os.path.basename(zip_file_path))[0]
    target_dir = os.path.join(savepath, base_name)
    tmp_dir = tempfile.mkdtemp()

    try:
        # Extract zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        # Find root directory
        extracted = os.listdir(tmp_dir)
        root = os.path.join(tmp_dir, extracted[0]) if len(extracted) == 1 and os.path.isdir(os.path.join(tmp_dir, extracted[0])) else tmp_dir

        # Validate directory structure
        required_dirs = ["catalogs", "dialogues", "figures", "pyfiles"]
        for d in required_dirs:
            if not os.path.isdir(os.path.join(root, d)):
                return "Invalid directory structure", False

        # Validate required files
        required_files = ["code.json", "message.json", "sort.json"]
        for f in required_files:
            if not os.path.isfile(os.path.join(root, "dialogues", f)):
                return "Missing required dialogue files", False

        # Move to target directory
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(root, target_dir)

        global dialogue_path
        dialogue_path = target_dir
        return "Dialogue imported successfully", True

    except Exception as e:
        return f"Import failed: {str(e)}", False
    finally:
        shutil.rmtree(tmp_dir)

def update_chat():
    """Load existing dialogue into UI"""
    with open(os.path.join(dialogue_path, 'dialogues', 'message.json'), "r", encoding="utf-8") as f:
        dialogue = json.load(f)

    # Get latest files
    py_files = sorted(os.listdir(os.path.join(dialogue_path, 'pyfiles')), reverse=True)
    images = sorted(os.listdir(os.path.join(dialogue_path, 'figures')), reverse=True)
    catalogs = sorted(os.listdir(os.path.join(dialogue_path, 'catalogs')), reverse=True)

    # Load code
    code_content = None
    if py_files:
        with open(os.path.join(dialogue_path, 'pyfiles', py_files[0]), 'r', encoding='utf-8') as f:
            code_content = f.read()

    # Load images
    image_paths = [os.path.join(dialogue_path, 'figures', img) for img in images]

    # Load catalog
    catalog_df = None
    if catalogs:
        with open(os.path.join(dialogue_path, 'catalogs', catalogs[0]), "r", encoding="utf-8") as f:
            catalog_df = pd.DataFrame(json.load(f))

    return (
        dialogue,
        gr.Code(language="python", value=code_content),
        catalog_df,
        image_paths,
        False  # Reset update state
    )

def check_save_path(path):
    """Save path validate"""
    if not path or os.path.splitext(path)[1]:
        return False, "Invalid directory path"
        
    if os.path.exists(path):
        if not os.path.isdir(path) or not os.access(path, os.W_OK):
            return False, "Path is not a writable directory"
    else:
        parent = os.path.dirname(path) or "."
        if not os.path.exists(parent) or not os.access(parent, os.W_OK):
            return False, "Parent directory not writable"
        try:
            os.makedirs(path, exist_ok=True)
            os.chmod(path, 0o777)
        except:
            return False, "Could not create directory"
    return True, "Path updated successfully"

def change_save_path(save_path, savepath):
    """Update save path"""
    state, info = check_save_path(save_path)
    if state:
        return gr.update(value=save_path), info
    else:
        return savepath, info
    
def visibility_three():
    """Section visibility toggles"""
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

def on_model_mode_change(model_mode, remote_model_name, openai_model):
    """Change the visible of model settings"""
    if model_mode == "local":
        return_state = (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    else:
        return_state = (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )
        return_state += on_remote_model_change(remote_model_name, openai_model)
        return_state += (gr.update(visible=True),)
    return return_state

def on_remote_model_change(remote_model_name, openai_model):
    """Change the visible of remote model settings"""
    if remote_model_name == "deepseek":
        return_state = (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif remote_model_name == "openai":
        return_state = (
            gr.update(visible=False),
            gr.update(visible=True),
        )
        return_state += (on_openai_model_change(openai_model),)
    return return_state

def on_openai_model_change(openai_model):
    """Change the visible of other model"""
    if openai_model == "other model":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
    
def nothinghappen(varible):
    """Nothing happen"""
    pass

def check_cic_state(msg):
    """Return 'Execution result:' then update output displays"""
    try:
        if "Execution result:" in msg[-1]['content']:
            return True
        return False
    except Exception:
        return False
    
def check_dia_state(msg):
    """Clear the chatbot then clear output displays"""
    try:
        if msg == []:
            return True
        return False
    except Exception:
        return False

def launch():
    """Main application setup with Gradio interface"""
    with gr.Blocks() as demo:
        gr.Markdown("# GWToolbox Assistant")

        # Control buttons
        with gr.Row():
            settings_btn = gr.Button("Model Settings")
            save_btn = gr.Button("Change Save Path")
            upload_btn = gr.Button("Import Dialogue")

        # Model settings section
        settings_available = gr.State(False)
        with gr.Column(visible=True) as model_settings:
            with gr.Row():
                model_mode = gr.Radio(["local", "remote"], label="Model Mode", value="remote")
                
                # Local model options
                local_model_name = gr.Radio(["Qwen", "chatglm"], label="Model Name", value="Qwen", visible=False)
                use_gpu_flag = gr.Checkbox(label="Use GPU", value=True, visible=False)
                
                # Remote model options
                remote_model_name = gr.Radio(["openai", "deepseek"], label="Remote Model", value="deepseek", visible=True)
                deepseek_model = gr.Radio(
                    ['deepseek-chat', 'deepseek-reasoner'],
                    label="DeepSeek Models (see https://api-docs.deepseek.com/)",
                    value="deepseek-chat",
                    visible=True
                )
                openai_model = gr.Radio(
                    ['gpt-4o', 'gpt-4.1', 'o4-mini', 'other model'],
                    label="OpenAI Models (see https://platform.openai.com/docs/models)",
                    value="gpt-4o",
                    visible=False
                )
                other_model = gr.Textbox(label='Other OpenAI Model', visible=False)
                model_set = gr.Button("Configure Model")
            
            api_key = gr.Textbox(label="API Key", visible=True)
            python_path_1 = gr.Textbox(label="GWToolbox Python Path ('python' uses current interpreter)",visible=True)
            python_path_2 = gr.Textbox(label="HEToolbox Python Path ('python' uses current interpreter)",visible=True)
            model_state = gr.Markdown()

        # Save path settings
        with gr.Column(visible=False) as change_savepath:
            with gr.Row(visible=True):
                save_path_input = gr.Textbox(label="Dialogue Save Path", value=os.path.join(current_dir, "record"), scale=3)
                savepath_state = gr.State(os.path.join(current_dir, "record"))
                change_savepath_btn = gr.Button("Confirm", scale=1)
            save_info = gr.Markdown()
            # gr.Markdown("You are not allowed to change save path")

        # Dialogue import
        with gr.Column(visible=False) as upload_dialogue:
            file_input = gr.File(file_types=[".zip"], label="Upload Dialogue Archive (.zip)")
            load_button = gr.Button("Load Dialogue")
            load_info = gr.Markdown()
            update_state = gr.State(False)

        # Main chat interface
        ui_update_state = gr.State(False)
        ui_clear_state = gr.State(False)
        chatbot = gr.Chatbot(value=[], type="messages", visible=True)
        chat_interface = gr.ChatInterface(
            fn=gradio_run,
            additional_inputs=[settings_available, savepath_state, python_path_1, python_path_2],
            chatbot=chatbot,
            type="messages"
        )

        # Output displays
        with gr.Row():
            code_btn = gr.Button("Code")
            catalog_btn = gr.Button("Catalog")
            image_btn = gr.Button("Images")
        code_display = gr.Code(visible=True)
        image_display = gr.Gallery(visible=False)
        catalog_display = gr.Dataframe(visible=False, interactive=False)

        # Button set
        settings_btn.click(
            visibility_three,
            outputs=[model_settings, change_savepath, upload_dialogue]
        )
        save_btn.click(
            visibility_three,
            outputs=[change_savepath, model_settings, upload_dialogue]
        )
        upload_btn.click(
            visibility_three,
            outputs=[upload_dialogue, model_settings, change_savepath]
        )
        code_btn.click(
            visibility_three,
            outputs=[code_display,catalog_display,image_display]
        )
        catalog_btn.click(
            visibility_three,
            outputs=[catalog_display,code_display,image_display]
        )
        image_btn.click(
            visibility_three,
            outputs=[image_display,code_display,catalog_display]
        )

        change_savepath_btn.click(
            change_save_path,
            inputs=[save_path_input, savepath_state],
            outputs=[savepath_state, save_info]
        )

        load_button.click(
                fn=read_dialogue,
                inputs=[file_input, savepath_state],
                outputs=[load_info, update_state]
            )
        
        model_set.click(
            set_ai,
            inputs=[model_mode,local_model_name, use_gpu_flag,
                    remote_model_name, deepseek_model, openai_model,
                    other_model, api_key, python_path_1, python_path_2],
            outputs=[model_state, settings_available, api_key]
        )

        # State update
        update_state.change(
            fn=update_chat,
            outputs=[chatbot,code_display,catalog_display,image_display,update_state],
            preprocess=True
        )
            
        model_mode.change(
            fn=on_model_mode_change,
            inputs=[model_mode, remote_model_name, openai_model],
            outputs=[
                local_model_name, use_gpu_flag,
                remote_model_name, deepseek_model,
                openai_model, other_model, api_key
            ]
        )

        remote_model_name.change(
            fn=on_remote_model_change,
            inputs=[remote_model_name, openai_model],
            outputs=[deepseek_model, openai_model, other_model]
        )

        openai_model.change(
            fn=on_openai_model_change,
            inputs=[openai_model],
            outputs=[other_model]
        )

        # I don't understand why but only to add this line that can change deepseek_model.
        deepseek_model.change(fn=nothinghappen,inputs=[deepseek_model],outputs=[])

        chat_interface.chatbot.change(
            fn=check_cic_state,
            inputs=[chat_interface.chatbot],
            outputs=[ui_update_state]
        )

        chat_interface.chatbot.change(
            fn=check_dia_state,
            inputs=[chat_interface.chatbot],
            outputs=[ui_clear_state]
        )

        ui_update_state.change(
            fn=update_ui,
            inputs=[code_display, catalog_display, image_display],
            outputs=[code_display, catalog_display, image_display],
            preprocess=True
        )

        ui_clear_state.change(
            fn=clear_ui,
            outputs=[code_display, catalog_display, image_display],
            preprocess=True
        )

    demo.launch(
                server_name="127.0.0.1",
                server_port=7860,
                root_path="/gradio",
                # share=True,
                )

if __name__ == "__main__":
    launch()