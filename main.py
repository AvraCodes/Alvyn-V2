from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import google.generativeai as genai
import os
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from typing import Dict, Any, List
import re
import json
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pandas as pd
import subprocess
import openai
import logging
import tempfile
import signal
import sys
from pathlib import Path
from datetime import datetime
import shutil
import mimetypes
import traceback

load_dotenv()

# Initialize Gemini model globally
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-pro")

# Progress tracking for long-running tasks
progress_tracker = {}

# File upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def setup_signal_handlers():
    """Setup graceful shutdown handlers."""
    def signal_handler(signum, frame):
        print(f"\n[SHUTDOWN] Received signal {signum}. Cleaning up...")
        # Save any in-progress work if needed
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# Initialize signal handlers
setup_signal_handlers()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging to a file
logging.basicConfig(
    filename="agent_logs.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def get_relevant_data(file_name: str, css_selector: str = None) -> Dict[str, Any]:
    """Extract data from HTML file using BeautifulSoup."""
    with open(file_name, encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    
    if css_selector:
        elements = soup.select(css_selector)
        return {"data": [el.get_text(strip=True) for el in elements]}
    return {"data": soup.get_text(strip=True)}

async def scrape_website(url: str) -> str:
    """Scrape website content using Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            content = await page.content()
            # Save content to file
            with open("scraped_content.html", "w", encoding="utf-8") as file:
                file.write(content)
            return content
        except Exception as e:
            return f"Error scraping website: {str(e)}"
        finally:
            await browser.close()

async def scrape_specific_data(url: str, selector: str = None) -> Dict[str, Any]:
    """
    Scrape specific data from a website using selectors.
    Args:
        url: Website URL to scrape
        selector: CSS selector to target specific elements (e.g., "div.article", "h1.title")
    Returns:
        Dictionary containing scraped data
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            # Navigate to the page and wait for content
            await page.goto(url, wait_until="networkidle")
            
            # If selector is provided, wait for it and get specific content
            if selector:
                await page.wait_for_selector(selector)
                elements = await page.query_selector_all(selector)
                data = []
                for element in elements:
                    text = await element.text_content()
                    data.append(text.strip())
                return {"data": data, "url": url}
            
            # If no selector, get full page content
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            return {"data": soup.get_text(strip=True), "url": url}
            
        except Exception as e:
            return {"error": f"Failed to scrape {url}: {str(e)}"}
        finally:
            await browser.close()

def task_breakdown(task: str):
    """Breaks down a task into smaller programmable steps using Google GenAI."""
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-pro")
        
        # Read prompt template
        try:
            with open('prompt.txt', 'r') as f:
                task_breakdown_prompt = f.read()
        except FileNotFoundError:
            task_breakdown_prompt = """Break down this task into clear steps:

When parsing tables:
- Use BeautifulSoup + pandas.read_html with StringIO.
- If the DataFrame has a MultiIndex as columns, flatten it:
    df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
- Use partial/lowercase substring matching to locate relevant columns.
    Example:
    for col in df.columns:
        if "population" in col.lower() and "density" not in col.lower():
            population_col = col
            break
- If the column isn’t found, print: f"Available columns: {df.columns}" and raise a clear error.
"""
        
        prompt = f"{task_breakdown_prompt}\n{task}"
        response = model.generate_content(prompt)
        
        # Save for debugging
        with open('broken_task.txt', 'w') as f:
            f.write(response.text)
        
        return response.text
    except Exception as e:
        return f"Error in task breakdown: {str(e)}"

async def scrape_with_query(url: str, query: str) -> Dict[str, Any]:
    """
    Scrape website based on a natural language query.
    Args:
        url: Website URL to scrape
        query: Natural language query (e.g., "find all article titles", "get contact information")
    Returns:
        Dictionary containing scraped data
    """
    # Configure Gemini to help determine selectors
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    try:
        # Get CSS selector from query using Gemini
        prompt = f"""
        Convert this query into appropriate CSS selectors for web scraping:
        Query: {query}
        Website: {url}
        Return only the CSS selector without explanation.
        """
        response = model.generate_content(prompt)
        selector = response.text.strip()
        
        # Use the generated selector to scrape
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="networkidle")
                await page.wait_for_selector(selector, timeout=5000)
                elements = await page.query_selector_all(selector)
                
                data = []
                for element in elements:
                    text = await element.text_content()
                    data.append(text.strip())
                
                return {
                    "query": query,
                    "url": url,
                    "selector_used": selector,
                    "data": data
                }
            
            except Exception as e:
                return {"error": f"Failed to scrape: {str(e)}"}
            finally:
                await browser.close()
                
    except Exception as e:
        return {"error": f"Failed to process query: {str(e)}"}

async def process_query_file(file_content: str, url: str) -> Dict[str, Any]:
    """
    Process multiple queries from a text file for a given URL.
    Each line in the file is treated as a separate query.
    
    Args:
        file_content: Content of the text file with queries
        url: Website URL to scrape
    Returns:
        Dictionary containing results for each query
    """
    queries = [q.strip() for q in file_content.splitlines() if q.strip()]
    results = []
    
    for query in queries:
        result = await scrape_with_query(url, query)
        results.append({
            "query": query,
            "result": result
        })
    
    return {
        "url": url,
        "total_queries": len(queries),
        "results": results
    }

@app.get("/")
async def root():
    return {"message": "Hello!"}

@app.post("/api/scrape")
async def scrape_url(url: str):
    content = await scrape_website(url)
    return {"content": get_relevant_data("scraped_content.html")}

@app.post("/api/process")
async def process_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.txt'):
        return {"error": "Only .txt files are supported"}
    contents = await file.read()
    text_content = contents.decode('utf-8')
    result = task_breakdown(text_content)
    return {
        "file_name": file.filename,
        "result": result,
        "message": "File processed successfully!"
    }

@app.post("/api/scrape-data")
async def scrape_endpoint(url: str, selector: str = None):
    """API endpoint to scrape specific data from a website"""
    result = await scrape_specific_data(url, selector)
    return result

@app.post("/api/smart-scrape")
async def smart_scrape_endpoint(url: str, query: str):
    """API endpoint that accepts a URL and natural language query"""
    result = await scrape_with_query(url, query)
    return result

@app.post("/api/bulk-scrape")
async def bulk_scrape_endpoint(
    file: UploadFile = File(...),
    url: str = None
):
    """API endpoint that accepts a text file with multiple queries"""
    if not file.filename.endswith('.txt'):
        return {"error": "Only .txt files are supported"}
    if not url:
        return {"error": "URL parameter is required"}
    
    try:
        contents = await file.read()
        text_content = contents.decode('utf-8')
        result = await process_query_file(text_content, url)
        return {
            "file_name": file.filename,
            "url": url,
            "results": result,
            "message": "Queries processed successfully!"
        }
    except Exception as e:
        return {"error": f"Failed to process queries: {str(e)}"}

MAX_RETRIES = 3  # Prevent infinite loops

@app.post("/api/")
async def universal_question_endpoint(file: UploadFile = File(...)):
    question_content = (await file.read()).decode("utf-8")
    with open("prompt.txt", "r") as f:
        prompt_content = f.read()
    llm_prompt = prompt_content + "\n\n" + question_content

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-pro")

    error_message = None
    prev_results = None

    for attempt in range(MAX_RETRIES):
        prompt_to_send = llm_prompt
        if error_message:
            prompt_to_send += (
                f"\n\nThe previous code failed with this error:\n{error_message}\n"
                "Please fix the code and try again. Only output a complete, corrected Python script."
            )

        response = model.generate_content(prompt_to_send)
        response_text = response.text
        code = extract_code_from_response(response_text)
        logging.info(f"Generated code (attempt {attempt+1}):\n{code}")
        with open("test_scraper.py", "w") as f:
            f.write(code)
        result = subprocess.run(
            ["python", "test_scraper.py"],
            capture_output=True, text=True, timeout=180
        )
        output = result.stdout.strip()
        if not output:
            output = result.stderr.strip()
        try:
            answers = json.loads(output)
            return answers  # Success! Return immediately without validation
        except Exception:
            error_message = output
            logging.error(f"Attempt {attempt+1} failed. Error:\n{error_message}\n")

    logging.error(f"Failed after {MAX_RETRIES} attempts. Last error: {error_message}")
    return {"error": f"Failed after {MAX_RETRIES} attempts. Last error: {error_message}"}

def extract_code_from_response(response_text):
    match = re.search(r"```(?:python)?(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()

async def process_universal_question(url: str, questions: list):
    """
    Scrape the site and answer the questions.
    """
    # --- Scrape the site ---
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        html = await page.content()
        await browser.close()
    soup = BeautifulSoup(html, "html.parser")

def answer_questions_from_file(filepath: str, max_retries: int = 3):
    """
    Reads questions from a .txt file, tries to answer each using Gemini (web/LLM),
    and if needed, generates and executes code to answer, returning a JSON array of answers.
    Enhanced with iterative error fixing.
    """
    try:
        with open(filepath, "r") as f:
            questions = [q.strip() for q in f.readlines() if q.strip()]
        if not questions:
            return ["No questions found in the file."]
        answers = []
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-pro")

        for i, question in enumerate(questions):
            logging.info(f"Processing question {i+1}: {question}")

            # Track progress
            progress_key = f"question_{i+1}"
            progress_tracker[progress_key] = {"status": "processing", "question": question}

            direct_prompt = f"""
Question: {question}

Can you answer this question directly using your knowledge or general web information?
If you can provide a confident, accurate answer without needing to scrape specific websites,
data analysis, or custom code execution, please provide the answer.

If this question requires:
- Scraping specific websites for current data
- Numerical analysis or calculations on scraped data
- Creating visualizations/plots
- Complex data processing

Then respond with exactly: "CODE_REQUIRED"

Otherwise, provide your direct answer.
"""
            try:
                response = model.generate_content(direct_prompt)
                direct_answer = response.text.strip()
                if "CODE_REQUIRED" not in direct_answer:
                    answers.append(direct_answer)
                    logging.info(f"Question {i+1} answered directly")
                    progress_tracker[progress_key] = {"status": "completed", "answer": direct_answer}
                    continue

                # Code generation with iterative fixing
                logging.info(f"Question {i+1} requires code generation")
                final_answer = generate_and_execute_code_with_retries(question, max_retries)
                answers.append(final_answer)
                progress_tracker[progress_key] = {"status": "completed", "answer": final_answer}

            except Exception as e:
                error_msg = f"Error processing question '{question}': {str(e)}"
                answers.append(error_msg)
                logging.error(error_msg)
                progress_tracker[progress_key] = {"status": "error", "error": error_msg}

        return answers
    except FileNotFoundError:
        return ["questions.txt file not found"]
    except Exception as e:
        return [f"Failed to process questions: {str(e)}"]

def generate_and_execute_code_with_retries(question: str, max_retries: int = 3) -> str:
    """Generate and execute code with retry logic for failed executions."""
    prev_result = None
    error_message = None

    for attempt in range(max_retries):
        code_prompt = f"""
You need to write a complete Python script to answer this question:
{question}

Requirements:
- Write a complete, standalone Python script
- Include all necessary imports
- Handle errors gracefully
- At the end, print the answer as a JSON array of strings using:
  import json
  print(json.dumps([answer]))
- If creating plots, encode as base64 PNG data URI under 100,000 bytes
- Make the code robust and handle edge cases
{f"- Previous attempt failed with error: {error_message}" if error_message else ""}

Generate the complete Python script:
"""

        try:
            code_response = model.generate_content(code_prompt)
            code = extract_code_from_response(code_response.text)
            if not code:
                return f"Error: Could not generate code for question: {question}"

            logging.info(f"Generated code for attempt {attempt+1}:\n{code}")

            # Execute the generated code
            temp_file = f"temp_question_attempt_{attempt+1}.py"
            with open(temp_file, "w") as f:
                f.write(code)

            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=180
            )

            output = result.stdout.strip()
            if not output:
                output = result.stderr.strip()

            try:
                code_answers = json.loads(output)
                if isinstance(code_answers, list) and len(code_answers) > 0:
                    final_answer = str(code_answers[0])
                else:
                    final_answer = str(code_answers)

                logging.info(f"Question answered successfully on attempt {attempt+1}")
                try:
                    os.remove(temp_file)
                except:
                    pass
                return final_answer

            except json.JSONDecodeError:
                error_message = f"Invalid JSON output: {output[:200]}"
                logging.warning(f"Attempt {attempt+1} produced invalid JSON: {error_message}")

        except subprocess.TimeoutExpired:
            error_message = "Code execution timed out"
            logging.error(f"Attempt {attempt+1} timed out")
        except Exception as e:
            error_message = str(e)
            logging.error(f"Attempt {attempt+1} failed with error: {error_message}")
        finally:
            # Clean up temp file
            try:
                if 'temp_file' in locals():
                    os.remove(temp_file)
            except:
                pass

    # If all attempts failed, return the last error
    return f"Failed after {max_retries} attempts. Last error: {error_message}"

@app.post("/api/auto-answer")
async def auto_answer_endpoint():
    """
    Automatically answers questions from questions.txt file with enhanced error handling
    """
    try:
        answers = answer_questions_from_file("questions.txt")
        return {
            "answers": answers,
            "message": "Questions processed successfully!",
            "progress": progress_tracker
        }
    except Exception as e:
        return {"error": f"Failed to process questions: {str(e)}"}

@app.get("/api/progress")
async def get_progress():
    """Get progress of currently running tasks."""
    return {"progress": progress_tracker, "timestamp": datetime.now().isoformat()}

@app.post("/api/clear-progress")
async def clear_progress():
    """Clear progress tracking data."""
    global progress_tracker
    progress_tracker = {}
    return {"message": "Progress cleared", "timestamp": datetime.now().isoformat()}

# File Management Endpoints
@app.post("/api/upload-files")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """Upload multiple files and store them in the uploads directory."""
    uploaded_files = []
    
    for file in files:
        try:
            # Generate unique filename to avoid conflicts
            file_extension = Path(file.filename).suffix
            unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            file_path = UPLOAD_DIR / unique_filename
            
            # Save file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Get file info
            file_size = len(content)
            mime_type, _ = mimetypes.guess_type(file.filename)
            
            uploaded_files.append({
                "original_name": file.filename,
                "stored_name": unique_filename,
                "size": file_size,
                "mime_type": mime_type or "application/octet-stream",
                "upload_time": datetime.now().isoformat()
            })
            
        except Exception as e:
            uploaded_files.append({
                "original_name": file.filename,
                "error": str(e)
            })
    
    return {
        "message": f"Uploaded {len([f for f in uploaded_files if 'error' not in f])} files successfully",
        "files": uploaded_files
    }

@app.get("/api/files")
async def list_uploaded_files():
    """List all uploaded files with metadata."""
    files_info = []
    
    try:
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                mime_type, _ = mimetypes.guess_type(file_path.name)
                
                # Extract original filename (remove timestamp prefix)
                original_name = file_path.name
                if "_" in original_name and len(original_name.split("_", 1)) > 1:
                    original_name = original_name.split("_", 1)[1]
                
                files_info.append({
                    "stored_name": file_path.name,
                    "original_name": original_name,
                    "size": stat.st_size,
                    "mime_type": mime_type or "application/octet-stream",
                    "upload_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "download_url": f"/api/files/{file_path.name}"
                })
        
        # Sort by upload time (newest first)
        files_info.sort(key=lambda x: x["upload_time"], reverse=True)
        
    except Exception as e:
        return {"error": f"Failed to list files: {str(e)}"}
    
    return {
        "total_files": len(files_info),
        "files": files_info
    }

@app.get("/api/files/{filename}")
async def download_file(filename: str):
    """Download a specific uploaded file."""
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        return {"error": "File not found"}
    
    return FileResponse(
        path=file_path,
        filename=filename.split("_", 1)[1] if "_" in filename else filename,  # Remove timestamp prefix
        media_type=mimetypes.guess_type(filename)[0] or "application/octet-stream"
    )

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """Delete a specific uploaded file."""
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        return {"error": "File not found"}
    
    try:
        file_path.unlink()
        return {"message": f"File {filename} deleted successfully"}
    except Exception as e:
        return {"error": f"Failed to delete file: {str(e)}"}

@app.delete("/api/files")
async def clear_all_files():
    """Delete all uploaded files."""
    try:
        deleted_count = 0
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                file_path.unlink()
                deleted_count += 1
        
        return {"message": f"Deleted {deleted_count} files successfully"}
    except Exception as e:
        return {"error": f"Failed to clear files: {str(e)}"}

@app.get("/files")
async def files_page():
    """Serve a simple HTML page to manage uploaded files."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Uploaded Files Manager</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .file-item { border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 5px; }
            .file-actions { margin-top: 10px; }
            button { background: #007bff; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; margin-right: 5px; }
            button:hover { background: #0056b3; }
            button.delete { background: #dc3545; }
            button.delete:hover { background: #c82333; }
            .upload-section { border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 5px; }
            .stats { background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>📁 Uploaded Files Manager</h1>
        
        <div class="upload-section">
            <h3>Upload Files</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="fileInput" multiple accept="*/*">
                <button type="submit">Upload Files</button>
            </form>
        </div>
        
        <div class="stats">
            <h3>Files Statistics</h3>
            <div id="stats">Loading...</div>
        </div>
        
        <div id="filesList">
            <h3>Uploaded Files</h3>
            <div id="files">Loading...</div>
        </div>
        
        <script>
            async function loadFiles() {
                try {
                    const response = await fetch('/api/files');
                    const data = await response.json();
                    
                    document.getElementById('stats').innerHTML = 
                        `<strong>Total Files:</strong> ${data.total_files}`;
                    
                    if (data.files && data.files.length > 0) {
                        const filesHtml = data.files.map(file => `
                            <div class="file-item">
                                <strong>${file.original_name}</strong><br>
                                <small>Size: ${(file.size / 1024).toFixed(2)} KB | 
                                Uploaded: ${new Date(file.upload_time).toLocaleString()}</small><br>
                                <div class="file-actions">
                                    <button onclick="downloadFile('${file.stored_name}')">Download</button>
                                    <button class="delete" onclick="deleteFile('${file.stored_name}')">Delete</button>
                                </div>
                            </div>
                        `).join('');
                        document.getElementById('files').innerHTML = filesHtml;
                    } else {
                        document.getElementById('files').innerHTML = '<p>No files uploaded yet.</p>';
                    }
                } catch (error) {
                    document.getElementById('files').innerHTML = '<p>Error loading files.</p>';
                }
            }
            
            async function uploadFiles() {
                const fileInput = document.getElementById('fileInput');
                const files = fileInput.files;
                
                if (files.length === 0) {
                    alert('Please select files to upload');
                    return;
                }
                
                const formData = new FormData();
                for (let file of files) {
                    formData.append('files', file);
                }
                
                try {
                    const response = await fetch('/api/upload-files', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    alert(result.message);
                    loadFiles(); // Refresh the list
                    fileInput.value = ''; // Clear file input
                } catch (error) {
                    alert('Upload failed: ' + error.message);
                }
            }
            
            async function downloadFile(filename) {
                window.open(`/api/files/${filename}`, '_blank');
            }
            
            async function deleteFile(filename) {
                if (confirm('Are you sure you want to delete this file?')) {
                    try {
                        const response = await fetch(`/api/files/${filename}`, {
                            method: 'DELETE'
                        });
                        const result = await response.json();
                        alert(result.message);
                        loadFiles(); // Refresh the list
                    } catch (error) {
                        alert('Delete failed: ' + error.message);
                    }
                }
            }
            
            // Handle form submission
            document.getElementById('uploadForm').addEventListener('submit', function(e) {
                e.preventDefault();
                uploadFiles();
            });
            
            // Load files on page load
            loadFiles();
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.post("/")
async def answer_questions(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temp location
        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        answers = answer_questions_from_file(tmp_path)
        os.remove(tmp_path)
        return JSONResponse(content=answers)
    except Exception as e:
        return JSONResponse(content=[f"Error: {str(e)}"], status_code=500)
