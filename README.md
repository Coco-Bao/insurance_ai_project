# Insurance AI Project

This project is an AI-powered insurance assistant built with Python and Docker. It integrates with Google’s generative language API and requires API keys to function properly.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Prerequisites](#prerequisites)  
- [Setup and Running](#setup-and-running)  
- [Using Environment Variables to Manage API Keys](#using-environment-variables-to-manage-api-keys)  
- [Docker Usage](#docker-usage)  
- [Verifying Environment Variables](#verifying-environment-variables)  
- [Notes](#notes)  
- [License](#license)  

---

## Project Overview

This project provides an AI insurance agent that leverages Google’s generative language API. It uses Python 3.9 slim as the base image and requires several system dependencies to support libraries such as `pdf2image`, `opencv`, and `pillow_heif`.

---

## Prerequisites

- Docker installed on your machine  
- Google API key with access to the Generative Language API  
- Basic familiarity with command line and Docker  

---

## Setup and Running

1. **Clone the repository**

git clone https://github.com/your-username/your-repo.git
cd your-repo

text

2. **Create a `.env` file**

See the next section for details on how to create and configure `.env`.

3. **Build the Docker image**

docker build -t insurance_ai_project .

text

4. **Run the Docker container**

Using `.env` file to pass environment variables (recommended):

docker run --env-file .env -p 5000:5000 insurance_ai_project

text

---

## Using Environment Variables to Manage API Keys

This project requires sensitive API keys to run, such as your Google API key. To keep your keys secure and separate from the code, we use environment variables managed via a `.env` file.

### 1. Create a `.env` File

In the root directory of the project, create a file named `.env` with the following content:

OPENAI_API_KEY=your_google_api_key_here
OPENAI_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai

text

> **Important:** Replace `your_google_api_key_here` with your own Google API key.

### 2. Prevent `.env` from Being Committed to GitHub

This project includes `.env` in `.gitignore` to ensure your sensitive keys are never pushed to GitHub. Make sure your `.env` file is not tracked by git.

If you accidentally committed `.env`, remove it with:

git rm --cached .env
git commit -m "Remove .env file from repository"
git push

text

---

## Docker Usage

### Running with `.env` file

docker run --env-file .env -p 5000:5000 insurance_ai_project

text

- Loads environment variables from `.env` into the container.  
- Maps container port 5000 to local port 5000.

### Alternative: Passing environment variables directly

docker run -e OPENAI_API_KEY=your_google_api_key_here -e OPENAI_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai -p 5000:5000 insurance_ai_project

text

---

## Verifying Environment Variables

To verify that environment variables are correctly set inside the running container:

1. Find the container ID or name:

docker ps

text

2. Access the container shell:

docker exec -it <container_id_or_name> /bin/bash

text

3. Check environment variables:

echo $OPENAI_API_KEY
echo $OPENAI_API_BASE

text

---

## Notes

- Make sure your `.env` file is **never** committed to any public repository.  
- If you want to share your project, provide a `.env.example` file with placeholder values for users to copy and fill in their own keys.  
- Adjust the `CMD` in your Dockerfile if your app’s entry point changes.  
- You can use Docker Compose for easier multi-container management and environment variable handling.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*If you have any questions or issues, feel free to open an issue on GitHub or contact the maintainer.*