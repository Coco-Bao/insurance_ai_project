# insurance_ai_projectUsing Environment Variables to Manage API Keys
This project requires sensitive API keys to run, such as your Google API key. To keep your keys secure and separate from the code, we use environment variables managed via a .env file.

1. Create a .env File
In the root directory of the project, create a file named .env with the following content:

text
OPENAI_API_KEY=your_google_api_key_here
OPENAI_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai
Important: Replace your_google_api_key_here with your own Google API key.

2. Prevent .env from Being Committed to GitHub
This project includes .env in .gitignore to ensure your sensitive keys are never pushed to GitHub. Make sure your .env file is not tracked by git.

3. Running the Project with Docker
To run the project inside a Docker container and load your environment variables, use the following command from the project root:

bash
docker run --env-file .env -p 5000:5000 insurance_ai_project
The --env-file .env flag loads all variables defined in your .env file into the container environment.

The -p 5000:5000 flag maps the container's port 5000 to your local machine.

4. Alternative: Passing Environment Variables Directly
If you prefer not to use a .env file, you can pass environment variables directly in the docker run command:

bash
docker run -e OPENAI_API_KEY=your_google_api_key_here -e OPENAI_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai -p 5000:5000 insurance_ai_project
5. Verifying Environment Variables Inside the Container
To verify that the environment variables are correctly set inside the running container, you can execute:

bash
docker exec -it <container_id> /bin/bash
echo $OPENAI_API_KEY
echo $OPENAI_API_BASE
Replace <container_id> with your container’s ID or name.