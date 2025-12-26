from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os
import json
import subprocess
import re
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
import pandas as pd
import glob
import numpy as np
from io import BytesIO
import base64
import requests
import sys
import google.generativeai as genai

load_dotenv()

app = FastAPI(title="Data Analyst Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def detect_available_data_files():
    """Detect all available data files in the directory"""
    data_files = []
    
    # Common data file extensions
    extensions = ['*.csv', '*.json', '*.xlsx', '*.xls', '*.tsv', '*.txt', '*.parquet', '*.png', '*.jpg', '*.jpeg', '*.gif']
    
    for ext in extensions:
        files = glob.glob(ext)
        data_files.extend(files)
    
    return data_files

def analyze_file_structure(filename):
    """Analyze the structure of a data file"""
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
            return {
                'type': 'CSV',
                'columns': list(df.columns),
                'shape': f"{len(df)} rows x {len(df.columns)} columns",
                'sample_data': df.head(3).to_dict('records')
            }
        elif filename.endswith('.json'):
            with open(filename, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return {
                        'type': 'JSON Array',
                        'columns': list(data[0].keys()) if isinstance(data[0], dict) else 'N/A',
                        'shape': f"{len(data)} records",
                        'sample_data': data[:3]
                    }
                else:
                    return {
                        'type': 'JSON Object',
                        'structure': str(type(data)),
                        'sample_data': str(data)[:500] if len(str(data)) > 500 else data
                    }
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filename)
            return {
                'type': 'Excel',
                'columns': list(df.columns),
                'shape': f"{len(df)} rows x {len(df.columns)} columns",
                'sample_data': df.head(3).to_dict('records')
            }
        elif filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            return {
                'type': 'Image',
                'filename': filename,
                'description': f"Image file: {filename}"
            }
        elif filename.endswith('.txt'):
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                return {
                    'type': 'Text',
                    'size': f"{len(content)} characters",
                    'sample_content': content[:300] if len(content) > 300 else content
                }
        else:
            return {'type': 'Unknown', 'filename': filename}
    except Exception as e:
        return {'type': 'Error', 'error': str(e), 'filename': filename}

def extract_code_from_response(response_text: str) -> str:
    """Extract Python code from LLM response and clean it"""
    if not response_text:
        return None
    
    # First try to extract from markdown code blocks
    patterns = [
        r'```python\s*\n(.*?)```',
        r'```\s*\n(.*?)```',
        r'```(.*?)```'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if any(keyword in code for keyword in ['import', 'def', 'print', 'json', 'results']):
                return code
    
    # If no code blocks found, check if the response is already raw code
    # Remove any leading/trailing backticks
    cleaned = response_text.strip()
    if cleaned.startswith('```python'):
        cleaned = cleaned[9:]
    elif cleaned.startswith('```'):
        cleaned = cleaned[3:]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # Check if it looks like Python code
    if any(keyword in cleaned for keyword in ['import', 'def', 'print', 'json', 'results', '=']):
        return cleaned
    
    return None

def extract_failed_blocks(script_content: str, failed_keys: List[str]) -> Dict[str, str]:
    """Extract try/except code blocks for the failed keys from the script."""
    blocks = {}
    for key in failed_keys:
        # Look for try: ... results['key'] = ... except ... results['key_error'] = ...
        try_block_pattern = rf"try:[\s\S]*?results\s*\[\s*['\"]({key})['\"].*?except[\s\S]*?results\s*\[\s*['\"](\1_error)['\"]"
        matches = re.finditer(try_block_pattern, script_content)
        
        for match in matches:
            # Find the start of the try block
            try_start = match.start()
            # Backtrack to find the actual try statement
            lines_before = script_content[:try_start].split('\n')
            
            # Find the line with 'try:'
            for i in range(len(lines_before) - 1, -1, -1):
                if lines_before[i].strip().startswith('try:'):
                    try_line_start = len('\n'.join(lines_before[:i]))
                    break
            else:
                try_line_start = try_start
            
            # Find the end of the except block
            except_start = match.end()
            lines_after = script_content[except_start:].split('\n')
            
            except_end = except_start
            indent_level = len(script_content[try_line_start:].split('\n')[0]) - len(script_content[try_line_start:].split('\n')[0].lstrip())
            
            for i, line in enumerate(lines_after):
                if line.strip() and not line.startswith(' ' * (indent_level + 1)) and line.strip():
                    except_end = except_start + len('\n'.join(lines_after[:i]))
                    break
            
            if except_end > try_line_start:
                blocks[key] = script_content[try_line_start:except_end]
                break
        
        if key not in blocks:
            blocks[key] = f"# Could not extract code block for {key}"
    
    return blocks

def fix_script_with_llm(
    failed_blocks: Dict[str, str],
    error_keys: List[str],
    error_messages: Dict[str, str],
    passed_keys: List[str],
    question_content: str,
    is_last_attempt: bool = False
) -> str:
    """Use Gemini LLM to fix only failed keys iteratively"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            logging.error("GEMINI_API_KEY not found in environment variables")
            return ""
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        error_context = "\n".join([
            f"- {key}: {error_messages.get(key, 'Unknown error')}"
            for key in error_keys
        ])
        
        failed_code = "\n".join([
            f"# Code for {key}:\n{failed_blocks.get(key, '')}"
            for key in error_keys
        ])
        
        fix_prompt = f"""You are a data analysis expert. The following code blocks failed during execution.

Failed keys and errors:
{error_context}

Failed code blocks:
{failed_code}

Questions being answered:
{question_content}

Successfully analyzed keys (do NOT modify):
{', '.join(passed_keys)}

Fix ONLY the failed code blocks. Requirements:
- Wrap each fix in a try/except block with results['key_name'] = ... and results['key_name_error'] = str(e)
- Keep the structure: results = {{}}, then try/except for each key
- Return ONLY valid Python code
- Handle errors gracefully
- Convert numpy types before JSON serialization
- Output valid JSON at the end: print(json.dumps(results))

Generate the complete fixed Python script:
"""
        
        response = model.generate_content(fix_prompt)
        
        if response and response.text:
            fix_response = response.text.strip()
            fixed_code = extract_code_from_response(fix_response)
            return fixed_code if fixed_code else fix_response
        
        return ""
        
    except Exception as e:
        logging.error(f"Error fixing script with LLM: {str(e)}")
        return ""

def get_gemini_response(prompt: str) -> str:
    """Get response from Google Gemini API with proper error handling"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            logging.error("GEMINI_API_KEY not found in environment variables")
            return ""
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            logging.error("No response from Gemini")
            return ""
            
    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}")
        return ""

def json_serializable(obj):
    """Convert numpy/pandas types to JSON serializable types"""
    if hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serializable(item) for item in obj]
    else:
        return obj

def analyze_data(questions_content: str, max_retries: int = 3):
    """Main data analysis function - generates code with iterative per-key error fixing"""
    try:
        logging.info("Starting code generation for analysis")

        # Detect available files
        available_files = detect_available_data_files()

        # Get detailed file analysis for code generation
        file_analysis = {}
        for file in available_files:
            file_analysis[file] = analyze_file_structure(file)

        # Create detailed file descriptions
        file_descriptions = []
        for file, analysis in file_analysis.items():
            desc = f"- {file}: {analysis.get('type', 'Unknown')}"
            if 'shape' in analysis:
                desc += f" ({analysis['shape']})"
            if 'columns' in analysis:
                desc += f", columns: {analysis['columns']}"
            file_descriptions.append(desc)

        # Read system prompt
        try:
            with open("prompt.txt", "r") as f:
                system_prompt = f.read()
        except FileNotFoundError:
            system_prompt = """You are a data analyst agent. Generate Python code to analyze data and answer questions accurately."""

        # Determine output format from questions
        is_json_object = ("JSON object" in questions_content or
                         "json object" in questions_content.lower() or
                         "return a JSON object" in questions_content.lower())
        output_format = "JSON object" if is_json_object else "JSON array"

        # Generate initial script
        script_content = generate_analysis_script(questions_content, file_descriptions, file_analysis, system_prompt, output_format)

        # Execute with iterative fixing
        final_result = execute_with_iterative_fixing(script_content, questions_content, max_retries)

        # Convert any remaining numpy types
        safe_result = json_serializable(final_result)
        return safe_result

    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        return {"error": str(e)}

def generate_analysis_script(questions_content: str, file_descriptions: list, file_analysis: dict, system_prompt: str, output_format: str) -> str:
    """Generate the initial analysis script using LLM"""
    code_prompt = f"""
{system_prompt}

Available data files:
{chr(10).join(file_descriptions)}

Detailed file analysis:
{json.dumps(file_analysis, indent=2)}

Questions to answer:
{questions_content}

Generate a complete Python script that:
1. Loads ALL available data files automatically using the exact filenames shown above
2. Performs the requested analysis precisely
3. Outputs results as a {output_format} using: print(json.dumps(results))

CRITICAL REQUIREMENTS:
- ONLY use these standard libraries: pandas, numpy, json, matplotlib, seaborn, scipy, requests, base64, io, glob
- DO NOT use: networkx, cv2, opencv, sklearn, tensorflow, torch, or any other non-standard packages
- Handle different file formats (CSV, JSON, Excel using pandas)
- For plots: return base64 PNG data URI starting with "data:image/png;base64," under 100KB
- Use try/except blocks for EACH result key with results[key] = value and results[key_error] = str(error)
- Convert numpy types to Python native types before JSON serialization using .item(), int(), float(), or .tolist()
- For web scraping: use pandas.read_html() for tables or requests + BeautifulSoup
- ALWAYS output valid JSON at the end using: print(json.dumps(results))
- Handle missing data gracefully with defaults
- Use exact column names and file names as provided
- DO NOT include markdown code blocks or backticks in your output
- Output ONLY raw Python code

Example structure:
import pandas as pd
import numpy as np
import json

results = {{}}

try:
    # Load data
    df = pd.read_csv('filename.csv')
except Exception as e:
    results['load_error'] = str(e)

try:
    results['key1'] = some_value
except Exception as e:
    results['key1_error'] = str(e)

print(json.dumps(results))

Generate ONLY the Python code (no markdown, no backticks):
"""

    code_response = get_gemini_response(code_prompt)

    if not code_response:
        raise Exception("Could not generate analysis code")

    # Extract code
    code = extract_code_from_response(code_response)
    if not code:
        code = code_response

    return code

def execute_with_iterative_fixing(script_content: str, questions_content: str, max_retries: int):
    """Execute script with iterative per-key error fixing"""
    prev_results = None
    current_script = script_content

    for retry_count in range(max_retries):
        if retry_count >= max_retries:
            logging.error("Max retries reached. Unable to execute script successfully.")
            return prev_results if prev_results else {"error": "Max retries exceeded"}

        try:
            script_path = "llm-code.py" if retry_count == 0 else f"llm-code-retry-{retry_count}.py"
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(current_script)

            print(f"Created LLM script at: {os.path.abspath(script_path)} (Attempt {retry_count+1}/{max_retries})")

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=300
            )

            stdout_output = result.stdout.strip()
            stderr_output = result.stderr.strip()

            if result.returncode == 0:
                # Try to parse JSON from stdout
                output_json = parse_json_from_output(stdout_output)
                
                if output_json is None:
                    logging.error("Script did not return valid JSON")
                    if retry_count < max_retries - 1:
                        # Try to regenerate with error context
                        error_context = f"Script executed but returned invalid JSON. Output: {stdout_output[:500]}"
                        current_script = regenerate_script_with_error(questions_content, error_context)
                        if current_script:
                            continue
                    return {"error": "Invalid JSON output", "stdout": stdout_output, "stderr": stderr_output}

                # Merge with previous results if any
                merged_results = dict(prev_results) if prev_results else {}
                merged_results.update(output_json)

                # Find failed keys (ending with _error)
                failed_keys = [k for k in merged_results if k.endswith('_error') and merged_results[k]]
                passed_keys = [k for k in merged_results if not k.endswith('_error')]

                # If no failed keys or max retries, return merged results
                if not failed_keys or retry_count >= max_retries - 1:
                    return merged_results

                # Otherwise, fix only failed keys
                logging.info(f"Found {len(failed_keys)} failed keys, attempting to fix...")
                failed_blocks = extract_failed_blocks(current_script, failed_keys)
                fixed_code = fix_failed_keys_with_llm(
                    failed_blocks=failed_blocks,
                    error_keys=failed_keys,
                    error_messages={k: merged_results[k] for k in failed_keys},
                    passed_keys=passed_keys,
                    question_content=questions_content,
                    is_last_attempt=(retry_count == max_retries - 2)
                )

                if fixed_code:
                    current_script = fixed_code
                    prev_results = merged_results
                else:
                    return merged_results
            else:
                logging.error(f"Script execution failed: {stderr_output}")
                
                # Try to regenerate with error context
                if retry_count < max_retries - 1:
                    error_context = f"Script failed with error: {stderr_output[:500]}"
                    regenerated = regenerate_script_with_error(questions_content, error_context)
                    if regenerated:
                        current_script = regenerated
                        continue
                
                return prev_results if prev_results else {"error": stderr_output, "stdout": stdout_output}

        except subprocess.TimeoutExpired:
            logging.error("Script execution timed out")
            return prev_results if prev_results else {"error": "Script execution timed out"}
        except Exception as e:
            logging.error(f"Unexpected error executing script: {e}")
            return prev_results if prev_results else {"error": str(e)}

    return prev_results if prev_results else {"error": "All retry attempts failed"}

def parse_json_from_output(output: str) -> dict:
    """Try to parse JSON from script output, handling edge cases"""
    if not output:
        return None
    
    # Try direct parsing first
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in the output
    # Look for last { ... } block
    brace_count = 0
    json_start = -1
    json_end = -1
    
    for i, char in enumerate(output):
        if char == '{':
            if brace_count == 0:
                json_start = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                json_end = i + 1
    
    if json_start >= 0 and json_end > json_start:
        try:
            return json.loads(output[json_start:json_end])
        except json.JSONDecodeError:
            pass
    
    # Try each line
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    
    return None

def regenerate_script_with_error(questions_content: str, error_context: str) -> str:
    """Regenerate script after an error occurred"""
    prompt = f"""
The previous Python script failed with the following error:
{error_context}

Please generate a corrected Python script for this task:
{questions_content}

Requirements:
- Fix the error mentioned above
- Do not use external packages that may not be installed (avoid networkx, cv2, etc.)
- Use only standard libraries: pandas, numpy, json, matplotlib, seaborn, scipy, requests
- Handle missing data gracefully
- Output results as JSON using: print(json.dumps(results))
- Wrap each result key in its own try-except block

Generate ONLY the Python code with no markdown formatting:
"""
    
    response = get_gemini_response(prompt)
    if response:
        return extract_code_from_response(response) or response
    return None

def fix_failed_keys_with_llm(failed_blocks, error_keys, error_messages, passed_keys, question_content, is_last_attempt=False):
    """Use LLM to fix only the failed code blocks"""
    if not any(os.getenv(key) for key in ["OPENAI_API_KEY"]):
        return None

    # Prepare the code blocks for the prompt
    code_blocks_str = "\n\n".join([f"# Block for {key}:\n{failed_blocks[key]}" for key in error_keys])

    prompt = f"""
You are required to generate Python code that fixes only the failed parts of a previous script for the following task.

Question/Task:
{question_content}

The following keys in the results had errors (with their error messages):
{json.dumps(error_messages, indent=2)}

The following keys PASSED and do NOT need to be regenerated:
{json.dumps(passed_keys)}

The code blocks below correspond to the failed keys. Only these blocks need to be fixed:
{code_blocks_str}

STRICT REQUIREMENTS:
1. Your output must be a single, complete, and runnable Python code block that ONLY computes the failed keys. No notes, explanations, markdown, placeholders, or comments outside the code.
2. Handle missing data gracefully.
3. If a failed key requires a visualization, generate it as a base64 PNG image under 100kB.
4. Return results as a JSON object with EXACTLY the failed keys (and their _error keys if any error occurs).
5. CRITICAL: Each failed key in the JSON output MUST be implemented in its own logical block with its own try-except.
- Do NOT wrap multiple keys together under one try/except.
- Each answer/plot/analysis step should be protected by a small, localized try-except block.
6. Use mock/sample data if external APIs are not available.
7. Do NOT include any triple backticks (```) or markdown formatting.
8. The code must be ready to execute AS-IS without any modifications.
9. Always return ONLY valid Python code — no natural language text at all outside of comments inside the code.
10. CRITICAL: Match the exact response format and JSON keys requested in the original task, but only for the failed keys.
11. IMPORTANT: Load data files directly from the current working directory (e.g., open('user-posts.json'), pd.read_csv('network-connections.csv')) without path prefixes.
12. MANDATORY: At the end of your script, print the final results as JSON to standard output. Always include:
    print(json.dumps(results, indent=2))
13. If a public API exists for the requested data, use it. Otherwise:
* 1) Fetch the webpage HTML.
* 2) Convert the HTML to Markdown with html2text (include the complete content).
* 3) Send the complete Markdown to OpenAI to obtain structured JSON.
* 4) When calling OpenAI:
* Use client.chat.completions.create(model="gpt-4.1", ...).
* You must use gpt-4.1 model only.
* The messages must instruct the model to return only a single compact/minified JSON object with no spaces or line breaks, no markdown, no backticks, and no explanations; use exactly and only the required keys; use null for missing values; if extraction fails, return {{"error":"reason"}}; do not include any text before or after the JSON.
* 5) When scraping:
* Follow robots.txt and site TOS, use a polite User-Agent, add small randomized delays, avoid heavy/abusive requests, and never bypass CAPTCHAs, paywalls, or access controls.

14. Try-Except:
- Each failed key must have its own try-except.
- If one visualization or analysis fails, continue with the next.
- Add appropriate error messages to the results dictionary when operations fail.
- Example pattern:
results = {{}}
# failed_key_1
try:
    results['failed_key_1'] = ...
except Exception as e:
    results['failed_key_1_error'] = f"{{type(e).__name__}}, {{e}}"

FINAL INSTRUCTION:
Do not add any notes, disclaimers, or partial/incomplete indicators.
The output must be a single, complete, runnable Python code block only, with one try/except per failed key.
{"THIS IS THE FINAL ATTEMPT. If a key still fails, just return the error message in the <key>_error field." if is_last_attempt else ""}
"""

    try:
        response = get_gemini_response(prompt)
        if response:
            fixed_code = extract_code_from_response(response)
            return fixed_code if fixed_code else response
    except Exception as e:
        logging.error(f"Error fixing script with LLM: {e}")
        return None

@app.post("/")
async def data_analyst_post_endpoint(request: Request):
    """
    POST endpoint that always generates code for analysis
    """
    try:
        raw_body = await request.body()
        content_type = request.headers.get("content-type", "").lower()
        
        logging.info(f"Content-Type: {content_type}")
        logging.info(f"Raw body length: {len(raw_body)}")
        
        questions_content = ""
        uploaded_files = []
        
        # Handle different content types
        if "application/json" in content_type:
            try:
                body = await request.json()
                if body is not None:
                    questions_content = (body.get("questions", "") or 
                                       body.get("question", "") or 
                                       body.get("query", "") or 
                                       body.get("text", "") or
                                       body.get("prompt", "") or
                                       body.get("input", ""))
                    
                    if not questions_content and isinstance(body, dict):
                        vars_dict = body.get("vars", {})
                        if isinstance(vars_dict, dict):
                            questions_content = str(vars_dict.get("question", ""))
                        
                    if not questions_content:
                        questions_content = str(body)
            except Exception as json_e:
                logging.error(f"JSON parsing error: {json_e}")
                questions_content = raw_body.decode('utf-8', errors='ignore').strip()
                
        elif "multipart/form-data" in content_type:
            try:
                form = await request.form()
                if form is not None:
                    logging.info(f"Form keys: {list(form.keys())}")
                    
                    # Extract questions.txt
                    questions_file = form.get("questions.txt")
                    if questions_file and hasattr(questions_file, 'read'):
                        content = await questions_file.read()
                        questions_content = content.decode('utf-8', errors='ignore').strip()
                        logging.info(f"Found questions.txt file with {len(questions_content)} chars")
                    
                    if not questions_content:
                        questions_content = (form.get("questions") or 
                                           form.get("question") or 
                                           form.get("query") or 
                                           form.get("text") or
                                           form.get("prompt"))
                        if questions_content:
                            questions_content = str(questions_content)
                    
                    # Process ALL additional files (data.csv, image.png, etc.)
                    for key, file in form.items():
                        if key != "questions.txt" and hasattr(file, 'filename') and file.filename:
                            try:
                                content = await file.read()
                                with open(file.filename, "wb") as f:
                                    f.write(content)
                                uploaded_files.append(file.filename)
                                logging.info(f"Saved uploaded file: {file.filename} ({len(content)} bytes)")
                            except Exception as file_e:
                                logging.error(f"Error saving file {file.filename}: {file_e}")
                                
                else:
                    logging.warning("Form data is None")
                    
            except Exception as form_e:
                logging.error(f"Form parsing error: {form_e}")
                questions_content = raw_body.decode('utf-8', errors='ignore').strip()
                
        else:
            try:
                questions_content = raw_body.decode('utf-8', errors='ignore').strip()
            except Exception as decode_e:
                logging.error(f"Decode error: {decode_e}")
                questions_content = str(raw_body)
        
        if questions_content is None:
            questions_content = ""
            
        logging.info(f"Extracted questions: {str(questions_content)[:200]}...")
        logging.info(f"Uploaded files: {uploaded_files}")
        
        if not questions_content or not str(questions_content).strip():
            logging.warning("No questions content found")
            return JSONResponse(
                content={
                    "error": "No questions found in request", 
                    "content_type": content_type,
                    "body_length": len(raw_body),
                    "uploaded_files": uploaded_files
                }, 
                status_code=400
            )

        # Always use code generation approach
        result = analyze_data(str(questions_content), max_retries=3)
        
        # Ensure result is JSON serializable before returning
        safe_result = json_serializable(result)
        
        return JSONResponse(content=safe_result)

    except Exception as e:
        logging.error(f"POST endpoint error: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"error": f"Internal error: {str(e)}"}, 
            status_code=500
        )

# Health check / API info endpoint
@app.get("/")
async def api_info():
    """API health check and info"""
    return JSONResponse(content={
        "status": "ok",
        "name": "Alvyn Data Analyst API",
        "version": "1.0.0",
        "endpoints": {
            "POST /": "Submit analysis request with questions.txt and data files"
        }
    })