
# Cloud Python Runner Frontend

This repository contains the frontend code for a web-based control panel designed to trigger and monitor Python scripts running on a server. It is a stylish, interactive, and responsive single-page application built with React, TypeScript, and Tailwind CSS.

## Features

- **Modern UI**: A sleek, dark-themed interface for a professional look and feel.
- **Interactive Controls**: Easily run predefined Python scripts or workflows with the click of a button.
- **Real-time Feedback**: View the status (Idle, Running, Success, Error) and terminal-like output for each script execution, including step-by-step logs for multi-script workflows.
- **Responsive Design**: The interface is fully responsive and works on desktops, tablets, and mobile devices.
- **Containerized**: Comes with a `Dockerfile` for easy deployment using Docker.

---

## How It Works

This application is a **frontend only**. It is designed to communicate with a backend API that you need to build separately. The backend would be responsible for:

1.  Exposing API endpoints for each major workflow (e.g., `/api/run/data-process`, `/api/run/train-model`, etc.).
2.  Executing the corresponding sequence of Python scripts when an endpoint is called. Your backend should orchestrate these scripts in the correct order:
    -   **`POST /api/run/data-process`** should run:
        1.  `backend/z_latest_performance.py`
        2.  `backend/z_latest_team_scr_ref.py`
    -   **`POST /api/run/train-model`** should run:
        1.  `backend/adjust_team_perform.py`
        2.  `backend/set_train_data.py`
        3.  `backend/ml_models/data-eng.py`
        4.  `backend/ml_models/train_model_new.py`
        5.  `backend/ml_models/train_AH_model.py`
        6.  `backend/ml_models/train_goals_model.py`
    -   **`POST /api/run/predict-res`** should run:
        1.  `backend/ml_models/predict_fixtures.py`
        2.  `backend/ml_models/predict_AH_model.py`
        3.  `backend/ml_models/predict_goals.py`
    -   **`POST /api/run/send-telegram`** should run:
        1.  `backend/send_tele.py`
3.  Streaming or returning the `stdout` and `stderr` from the script execution back to the frontend.

The buttons in this UI are pre-configured to make `POST` requests to these endpoints. The output shown is currently mocked for demonstration purposes.

---

## Local Development Setup

To run this application on your local machine, you'll need [Node.js](https://nodejs.org/) (v18 or higher) and `npm`.

**1. Install Dependencies:**
Navigate to the project root and install the required packages.
```bash
npm install
```
*(You will need to create a `package.json` file. See Appendix A for an example.)*

**2. Start the Development Server:**
```bash
npm run dev
```
*(This command depends on having a tool like Vite or Create React App configured in your `package.json`)*

The application should now be running on `http://localhost:5173` (or another port specified by your development tool).

---

## Deployment Guide

This application is designed to be deployed as a static site. The recommended method is using Docker, which is ideal for services like AWS EC2.

### Option 1: Deploying with Docker (Recommended for EC2)

The included `Dockerfile` creates a lightweight, production-ready Nginx server to serve the application.

**1. Build the Docker Image:**
From the project root, run the following command:
```bash
docker build -t cloud-python-runner .
```

**2. Run the Docker Container:**
Once the image is built, you can run it as a container:
```bash
docker run -d -p 8080:80 --name python-runner-ui cloud-python-runner
```
- `-d`: Runs the container in detached mode.
- `-p 8080:80`: Maps port 8080 on your host machine to port 80 inside the container.
- `--name python-runner-ui`: Assigns a convenient name to the container.

The application will be accessible at `http://localhost:8080`.

**3. Uploading to a Server (like AWS EC2):**

*   **Push Image to a Registry:** The standard way to get your image onto an EC2 instance is to push it to a container registry like Amazon ECR (Elastic Container Registry) or Docker Hub.
    1.  Tag your image: `docker tag cloud-python-runner:latest YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cloud-python-runner:latest`
    2.  Push the image: `docker push YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cloud-python-runner:latest`
*   **Pull and Run on EC2:**
    1.  SSH into your EC2 instance (which must have Docker installed).
    2.  Authenticate Docker with your registry.
    3.  Pull the image: `docker pull YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cloud-python-runner:latest`
    4.  Run the container as shown in step 2.

### Option 2: Manual Static Build

If you prefer not to use Docker, you can build the static files and serve them with any web server (Nginx, Apache, etc.).

**1. Build the Application:**
```bash
npm run build
```
This command will compile the application into a `dist` (or `build`) folder.

**2. Upload and Serve:**
Upload the contents of the `dist` folder to your web server's root directory. Configure your server to serve `index.html` for any requests that don't match a file, which is standard for Single-Page Applications.

---

## Appendix A: Example `package.json`

You will need to create this file in your project's root directory for the `npm` and `Dockerfile` commands to work. This example uses **Vite** as the build tool, which is fast and modern.

```json
{
  "name": "cloud-python-runner",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.15",
    "@types/react-dom": "^18.2.7",
    "@vitejs/plugin-react": "^4.0.3",
    "typescript": "^5.0.2",
    "vite": "^4.4.5"
  }
}
```
To set up a project with this configuration, you would typically run `npm create vite@latest . -- --template react-ts`.