# pr-aut – Pull Request Automation with LLM

An intelligent automation tool that analyzes code changes and automatically creates GitHub pull requests with AI-generated titles and descriptions.

## **Features**

- **Automated PR Creation**: Automatically creates GitHub pull requests from your code changes
- **AI-Generated Content**: Uses Groq's LLM to generate meaningful PR titles and descriptions based on git diffs
- **Git Diff Analysis**: Intelligently analyzes file changes using multiple strategies (staged, working directory, HEAD comparison)
- **Content-Based Diff**: Fallback mechanism for generating diffs when git history isn't available
- **Flexible Input Methods**: Supports both command-line file input and JSON-based batch processing

## **Tech Stack**

- **Backend**: Python 3.12+
- **AWS Services**: DynamoDB for user management
- **AI/ML**: Groq API with Llama 3 8B model for intelligent PR content generation
- **Build Tools**: esbuild for TypeScript compilation
- **APIs**: GitHub REST API for repository operations

## **How It Works**

1. **Branch Creation**: Creates a new timestamped branch from the default branch
2. **File Upload**: Uploads your file changes to the new branch
3. **Diff Analysis**: Generates comprehensive git diffs using multiple strategies:
   - Staged changes (`git diff --cached`)
   - Working directory changes (`git diff`)
   - HEAD comparison (`git diff HEAD`)
   - Content-based diff as fallback
4. **AI Analysis**: Sends the diff to Groq's LLM for intelligent analysis
5. **PR Creation**: Creates a pull request with AI-generated title and description

## **Project Structure**

pr-aut/
├── main.py              # Main automation script
├── profile.ts           # TypeScript Lambda handler
├── index.js             # Compiled JavaScript (generated)
├── build.js             # esbuild configuration
├── package.json         # Node.js dependencies
├── pyproject.toml       # Python project configuration
└── .env                 # Environment variables (create this)


