import os
import requests
import datetime
import base64
import json
import sys
import subprocess
from typing import Optional
from pydantic import BaseModel, Field
from groq import Groq


def load_env_file():
    """Load environment variables from .env file"""
    env_paths = [
        "env.txt",  # Current directory
        os.path.join(os.path.dirname(__file__), ".env"),  # Same directory as script
        os.path.expanduser("~/Desktop/pr-aut/.env"),  # Your specific path
    ]

    for env_path in env_paths:
        if os.path.exists(env_path):
            print(f"Loading environment from: {env_path}")
            try:
                with open(env_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            os.environ[key] = value
                return True
            except Exception as e:
                print(f"Warning: Could not load .env file {env_path}: {e}")

    print("No .env file found. Using system environment variables.")
    return False


# Load environment variables
load_env_file()

# Configuration
GITHUB_API = "https://api.github.com"
REPO_OWNER = "shrutiashok7"
REPO_NAME = "dummy-pr-aut"
DEFAULT_BRANCH = "main"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"

# Environment variables (loaded from .env file or system)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate required environment variables
if not GITHUB_TOKEN:
    print("❌ ERROR: GITHUB_TOKEN not found!")
    print("Please check your .env file or set environment variable:")
    print("export GITHUB_TOKEN='your_token_here'")
    sys.exit(1)

if not GROQ_API_KEY:
    print("❌ ERROR: GROQ_API_KEY not found!")
    print("Please check your .env file or set environment variable:")
    print("export GROQ_API_KEY='your_key_here'")
    sys.exit(1)

print(f"✓ GitHub Token loaded: {GITHUB_TOKEN[:8]}...")
print(f"✓ Groq API Key loaded: {GROQ_API_KEY[:8]}...")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)


# Data models
class PRRequest(BaseModel):
    file_path: str = Field(..., description="Path to the file to be uploaded")
    file_content: str = Field(..., description="Content of the file")
    commit_message: Optional[str] = Field(
        default=None, description="Custom commit message"
    )


class PRResponse(BaseModel):
    title: str = Field(..., description="Generated PR title")
    description: str = Field(..., description="Generated PR description")
    branch_name: str = Field(..., description="Name of the created branch")
    pr_url: str = Field(..., description="URL of the created pull request")
    status: str = Field(..., description="Status of the operation")


class LLMResponse(BaseModel):
    title: str = Field(..., description="PR title")
    description: str = Field(..., description="PR description")


def get_git_diff_for_file(file_path: str, project_folder: str, max_chars: int = 12000) -> str:
    """Get git diff for a specific file using local git commands"""
    try:
        # Get the absolute path to the actual file
        actual_file_path = os.path.join(project_folder, file_path)

        # Change to the project directory to run git commands
        repo_dir = os.path.abspath(project_folder)

        print(f"DEBUG: Looking for changes in {actual_file_path}")
        print(f"DEBUG: Running git commands from {repo_dir}")

        # Check if we're in a git repository
        git_check = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
        )

        if git_check.returncode != 0:
            print("DEBUG: Not in a git repository, using content-based diff")
            return get_content_based_diff(actual_file_path, file_path)

        # Strategy 1: Check if file is staged
        result = subprocess.run(
            ["git", "diff", "--cached", "--", file_path],
            capture_output=True,
            text=True,
            cwd=repo_dir,
        )

        diff = result.stdout.strip()
        print(f"DEBUG: git diff --cached result: {len(diff)} chars")

        # Strategy 2: Check working directory changes
        if not diff:
            result = subprocess.run(
                ["git", "diff", "--", file_path],
                capture_output=True,
                text=True,
                cwd=repo_dir,
            )
            diff = result.stdout.strip()
            print(f"DEBUG: git diff (working dir) result: {len(diff)} chars")

        # Strategy 3: Compare with HEAD
        if not diff:
            result = subprocess.run(
                ["git", "diff", "HEAD", "--", file_path],
                capture_output=True,
                text=True,
                cwd=repo_dir,
            )
            diff = result.stdout.strip()
            print(f"DEBUG: git diff HEAD result: {len(diff)} chars")

        # Strategy 4: If still no diff, use content-based comparison
        if not diff:
            print("DEBUG: No git diff found, using content-based diff")
            diff = get_content_based_diff(actual_file_path, file_path)

        # If we have a proper git diff, use it
        if diff and diff.startswith("diff --git"):
            if len(diff) > max_chars:
                print(
                    f"Diff is too large ({len(diff)} chars). Truncating to {max_chars} chars."
                )
                diff = diff[:max_chars]

            print(f"✓ Generated git diff for {file_path}")
            print(f"DEBUG: Git diff preview: {diff[:200]}...")
            return diff

        # Otherwise, use the content-based diff
        content_diff = get_content_based_diff(actual_file_path, file_path)
        if content_diff and content_diff != f"Updated file: {file_path}":
            return content_diff

        print(f"⚠️  No meaningful changes found for {file_path}")
        return f"Updated file: {file_path}"

    except Exception as e:
        print(f"✗ Error getting git diff: {e}")
        # Fallback to content-based diff
        try:
            return get_content_based_diff(os.path.join(project_folder, file_path), file_path)
        except:
            return f"Updated file: {file_path}"


def get_content_based_diff(actual_file_path: str, repo_file_path: str) -> str:
    """Generate a detailed diff based on file content comparison"""
    try:
        print(f"DEBUG: Starting content-based diff for {repo_file_path}")

        # Get current file content from GitHub main branch
        current_content = get_current_file_content(repo_file_path)
        print(f"DEBUG: Retrieved {len(current_content)} chars from GitHub")

        # Get local file content
        if os.path.exists(actual_file_path):
            with open(actual_file_path, "r") as f:
                new_content = f.read()
            print(f"DEBUG: Retrieved {len(new_content)} chars from local file")
        else:
            print(f"DEBUG: File {actual_file_path} not found locally")
            return f"New file: {repo_file_path}"

        # If contents are identical, no changes
        if current_content == new_content:
            print("DEBUG: File contents are identical")
            return f"No changes detected in {repo_file_path}"

        # Generate detailed diff
        diff_result = generate_detailed_diff(
            current_content, new_content, repo_file_path
        )
        print(f"DEBUG: Generated diff with {len(diff_result)} chars")
        return diff_result

    except Exception as e:
        print(f"✗ Error generating content-based diff: {e}")
        return f"Updated file: {repo_file_path}"


def generate_detailed_diff(old_content: str, new_content: str, file_path: str) -> str:
    """Generate a detailed, line-by-line diff between old and new content"""
    old_lines = old_content.splitlines() if old_content else []
    new_lines = new_content.splitlines() if new_content else []

    print(f"DEBUG: Comparing {len(old_lines)} old lines vs {len(new_lines)} new lines")

    # Create a simple unified diff format
    diff_lines = []
    diff_lines.append(f"--- a/{file_path}")
    diff_lines.append(f"+++ b/{file_path}")

    # Find the differences using a simple approach
    max_lines = max(len(old_lines), len(new_lines))

    # Track changes
    changes_found = False
    context_lines = 3  # Number of context lines to show around changes

    i = 0
    while i < max_lines:
        old_line = old_lines[i] if i < len(old_lines) else None
        new_line = new_lines[i] if i < len(new_lines) else None

        # If lines are different, we found a change
        if old_line != new_line:
            changes_found = True

            # Add context before the change
            start_context = max(0, i - context_lines)
            diff_lines.append(
                f"@@ -{start_context + 1},{min(len(old_lines), i + context_lines + 1) - start_context} +{start_context + 1},{min(len(new_lines), i + context_lines + 1) - start_context} @@"
            )

            # Add context lines before
            for j in range(start_context, i):
                if j < len(old_lines):
                    diff_lines.append(f" {old_lines[j]}")

            # Add the changed line(s)
            if old_line is not None:
                diff_lines.append(f"-{old_line}")
            if new_line is not None:
                diff_lines.append(f"+{new_line}")

            # Add context lines after
            for j in range(i + 1, min(max_lines, i + context_lines + 1)):
                line_to_add = (
                    new_lines[j]
                    if j < len(new_lines)
                    else old_lines[j] if j < len(old_lines) else None
                )
                if line_to_add is not None:
                    diff_lines.append(f" {line_to_add}")

            # Skip ahead to avoid duplicate processing
            i += context_lines + 1
        else:
            i += 1

    if not changes_found:
        # Check if there are additions at the end
        if len(new_lines) > len(old_lines):
            changes_found = True
            added_lines = new_lines[len(old_lines) :]
            diff_lines.append(
                f"@@ -{len(old_lines)},0 +{len(old_lines) + 1},{len(added_lines)} @@"
            )
            for line in added_lines:
                diff_lines.append(f"+{line}")
        elif len(old_lines) > len(new_lines):
            changes_found = True
            removed_lines = old_lines[len(new_lines) :]
            diff_lines.append(
                f"@@ -{len(new_lines) + 1},{len(removed_lines)} +{len(new_lines)},0 @@"
            )
            for line in removed_lines:
                diff_lines.append(f"-{line}")

    if changes_found:
        result = "\n".join(diff_lines)
        print(f"DEBUG: Generated detailed diff:\n{result[:300]}...")
        return result

    return f"No changes detected in {file_path}"


def create_branch(branch: str) -> bool:
    """Create a new branch from the default branch"""
    try:
        # Get the SHA of the default branch
        url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/git/ref/heads/{DEFAULT_BRANCH}"
        response = requests.get(
            url, headers={"Authorization": f"Bearer {GITHUB_TOKEN}"}
        )

        if response.status_code != 200:
            raise Exception(
                f"Cannot get base SHA from branch '{DEFAULT_BRANCH}'. Status: {response.status_code}"
            )

        base_sha = response.json()["object"]["sha"]

        # Create new branch
        create_url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/git/refs"
        data = {"ref": f"refs/heads/{branch}", "sha": base_sha}

        response = requests.post(
            create_url, headers={"Authorization": f"Bearer {GITHUB_TOKEN}"}, json=data
        )

        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create branch: {response.text}")

        print(f"✓ Created branch: {branch}")
        return True

    except Exception as e:
        print(f"✗ Error creating branch: {e}")
        return False


def upload_file_content(
    branch: str, file_path: str, content: str, commit_message: Optional[str] = None
) -> bool:
    """Upload file content to GitHub repository"""
    try:
        # Encode content to base64
        encoded_content = base64.b64encode(content.encode()).decode()

        # API URL for the file
        url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{file_path}"

        # Check if file already exists to get SHA
        response = requests.get(
            url, headers={"Authorization": f"Bearer {GITHUB_TOKEN}"}
        )
        existing_sha = None

        if response.status_code == 200:
            existing_sha = response.json().get("sha")

        # Prepare commit data
        message = commit_message or f"Update {file_path}"
        data = {"message": message, "content": encoded_content, "branch": branch}

        if existing_sha:
            data["sha"] = existing_sha

        # Upload/update the file
        response = requests.put(
            url, headers={"Authorization": f"Bearer {GITHUB_TOKEN}"}, json=data
        )

        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to upload file: {response.text}")

        print(f"✓ Uploaded {file_path} to branch {branch}")
        return True

    except Exception as e:
        print(f"✗ Error uploading file: {e}")
        return False


def get_current_file_content(file_path: str) -> str:
    """Get current file content from main branch"""
    try:
        url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{file_path}"
        params = {"ref": DEFAULT_BRANCH}
        response = requests.get(
            url, headers={"Authorization": f"Bearer {GITHUB_TOKEN}"}, params=params
        )

        if response.status_code == 200:
            content = response.json().get("content", "")
            return base64.b64decode(content).decode()
        elif response.status_code == 404:
            # File doesn't exist in main branch
            return ""
        else:
            raise Exception(f"Failed to get current file content: {response.text}")

    except Exception as e:
        print(f"✗ Error getting current file content: {e}")
        return ""


def generate_simple_diff(old_content: str, new_content: str) -> str:
    """Generate a simple summary of changes (keeping the original function as backup)"""
    old_lines = old_content.splitlines() if old_content else []
    new_lines = new_content.splitlines() if new_content else []

    # Simple line-by-line comparison
    added_lines = []
    removed_lines = []

    # Find lines that were added (in new but not in old)
    old_set = set(old_lines)
    new_set = set(new_lines)

    for line in new_lines:
        if line.strip() and line not in old_set:
            added_lines.append(line.strip())

    for line in old_lines:
        if line.strip() and line not in new_set:
            removed_lines.append(line.strip())

    # Build summary
    changes = []

    if added_lines:
        changes.append("ADDED:")
        for line in added_lines[-5:]:  # Show last 5 additions
            changes.append(f"+ {line}")

    if removed_lines:
        changes.append("REMOVED:")
        for line in removed_lines[-5:]:  # Show last 5 removals
            changes.append(f"- {line}")

    if changes:
        return "\n".join(changes)

    return "Minor file updates"


def generate_pr_title_and_description_with_groq_client(diff_text: str) -> LLMResponse:
    """Generate PR title and description using Groq client"""
    try:
        system_prompt = """You are an expert technical writer. Based on the provided git diff, generate a concise GitHub Pull Request title and description.

Rules:
- Focus ONLY on the exact changes shown in the diff
- Title should be brief and relevant (max 50 characters)
- Description should be bullet-pointed and structured
- Do NOT use phrases like 'This PR introduces...'
- Be precise and informative about what was actually changed
- Only describe what was literally added, removed, or modified"""

        user_prompt = f"""Generate a PR title and description based on ONLY these exact changes:

{diff_text}

Respond only with valid JSON: {{"title": "brief title", "description": "structured description"}}"""

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=GROQ_MODEL,
            temperature=0.3,  # Slightly higher for more natural language
        )

        content = chat_completion.choices[0].message.content.strip()
        print(f"DEBUG: LLM Response: {content}")
        response_data = json.loads(content)
        return LLMResponse(**response_data)

    except Exception as e:
        raise Exception(f"Error with Groq client: {e}")


def generate_pr_title_and_description_with_api(diff_text: str) -> LLMResponse:
    """Generate PR title and description using direct API calls"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        system_prompt = """You are an expert technical writer. Based on the provided git diff, generate a concise GitHub Pull Request title and description.

Rules:
- Focus ONLY on the exact changes shown in the diff
- Title should be brief and relevant (max 50 characters)
- Description should be bullet-pointed and structured
- Do NOT use phrases like 'This PR introduces...'
- Be precise and informative about what was actually changed
- Only describe what was literally added, removed, or modified"""

        user_prompt = f"""Generate a PR title and description based on ONLY these exact changes:

{diff_text}

Respond only with valid JSON: {{"title": "brief title", "description": "structured description"}}"""

        data = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,  # Slightly higher for more natural language
        }

        response = requests.post(GROQ_API_URL, headers=headers, json=data)

        if response.status_code != 200:
            raise Exception(f"API call failed: {response.text}")

        content = response.json()["choices"][0]["message"]["content"].strip()
        print(f"DEBUG: LLM Response: {content}")
        response_data = json.loads(content)
        return LLMResponse(**response_data)

    except Exception as e:
        raise Exception(f"Error with API call: {e}")


def generate_pr_title_and_description(diff_text: str) -> LLMResponse:
    """Generate PR title and description using available method"""
    if not diff_text.strip():
        raise ValueError("No diff found. Make sure you have changes to commit.")

    # Try Groq client first, then fallback to direct API
    try:
        return generate_pr_title_and_description_with_groq_client(diff_text)
    except Exception as e:
        print(f"Groq client failed, trying direct API: {e}")
        return generate_pr_title_and_description_with_api(diff_text)


def create_pull_request(branch: str, title: str, body: str) -> str:
    """Create a pull request"""
    try:
        url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/pulls"
        data = {"title": title, "head": branch, "base": DEFAULT_BRANCH, "body": body}

        response = requests.post(
            url, headers={"Authorization": f"Bearer {GITHUB_TOKEN}"}, json=data
        )

        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create PR: {response.text}")

        pr_url = response.json()["html_url"]
        print(f"✓ Pull request created: {pr_url}")
        return pr_url

    except Exception as e:
        print(f"✗ Error creating pull request: {e}")
        raise


def read_file_content(file_path: str) -> str:
    """Read content from a local file"""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {e}")


def process_pr_request(request_data: dict, project_folder: str) -> PRResponse:
    """Main function to process PR creation request"""
    try:
        # Validate input
        pr_request = PRRequest(**request_data)

        # Generate branch name
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        branch = f"auto-pr-{timestamp}"

        # Step 1: Create branch
        if not create_branch(branch):
            raise Exception("Failed to create branch")

        # Step 2: Upload file content
        if not upload_file_content(
            branch,
            pr_request.file_path,
            pr_request.file_content,
            pr_request.commit_message,
        ):
            raise Exception("Failed to upload file")

        # Step 3: Get git diff for the specific file
        diff = get_git_diff_for_file(pr_request.file_path, project_folder)

        # Step 4: Generate PR metadata
        llm_response = generate_pr_title_and_description(diff)

        # Step 5: Create pull request
        pr_url = create_pull_request(
            branch, llm_response.title, llm_response.description
        )

        # Return response
        return PRResponse(
            title=llm_response.title,
            description=llm_response.description,
            branch_name=branch,
            pr_url=pr_url,
            status="success",
        )

    except Exception as e:
        print(f"✗ Error processing PR request: {e}")
        raise


def parse_arguments():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated PR creation")
    
    parser.add_argument(
        "--folder", 
        required=True,
        help="Path to the project folder containing the files to be processed"
    )
    parser.add_argument(
        "--file", 
        help="Specific file path relative to the folder (legacy mode)"
    )
    parser.add_argument(
        "--input", 
        help="JSON input file or JSON string containing file_path, file_content, and optional commit_message"
    )
    parser.add_argument(
        "--output", 
        help="Output file for JSON response (optional)"
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Validate that folder exists
    if not os.path.exists(args.folder):
        print(f"❌ ERROR: Folder {args.folder} does not exist!")
        sys.exit(1)
    
    project_folder = os.path.abspath(args.folder)
    print(f"✓ Using project folder: {project_folder}")

    try:
        # Legacy mode: --file argument
        if args.file:
            print("Running in legacy mode...")

            # Full path to the file
            full_file_path = os.path.join(project_folder, args.file)
            
            if not os.path.exists(full_file_path):
                print(f"❌ ERROR: File {full_file_path} does not exist!")
                sys.exit(1)

            # Read file content
            file_content = read_file_content(full_file_path)

            # Create request data
            input_data = {
                "file_path": args.file,
                "file_content": file_content,
                "commit_message": f"Update {args.file}",
            }

        # New mode: --input argument
        elif args.input:
            print("Running in JSON mode...")

            if os.path.isfile(args.input):
                with open(args.input, "r") as f:
                    input_data = json.load(f)
            else:
                input_data = json.loads(args.input)

        else:
            print("Error: Please provide either --file or --input argument")
            print("\nExamples:")
            print("  python main.py --folder /path/to/project --file app/calculator.py")
            print("  python main.py --folder /path/to/project --input input.json")
            print('  python main.py --folder /path/to/project --input \'{"file_path": "app/test.py", "file_content": "print(\\"hello\\")"}\' ')
            sys.exit(1)

        # Process request
        result = process_pr_request(input_data, project_folder)

        # Output result
        output_json = result.model_dump_json(indent=2)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output_json)
            print(f"✓ Results saved to {args.output}")
        else:
            print("\n" + "=" * 50)
            print("PR CREATION RESULT:")
            print("=" * 50)
            print(output_json)

    except Exception as e:
        error_response = {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat(),
        }

        error_json = json.dumps(error_response, indent=2)

        if args.output:
            with open(args.output, "w") as f:
                f.write(error_json)
        else:
            print("ERROR:")
            print(error_json)

        sys.exit(1)


if __name__ == "__main__":
    main()