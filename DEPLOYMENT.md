# Deployment Guide

This guide covers deploying AdvisoryAI Jarvis to various hosting platforms.

## 🚀 Streamlit Cloud (Recommended - Free)

Streamlit Cloud is the easiest and free option for deploying Streamlit apps.

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at share.streamlit.io)
- Azure OpenAI credentials

### Steps

1. **Prepare your repository**
   - Ensure all credentials are removed from code
   - Verify `.env` is in `.gitignore`
   - Push code to a public GitHub repository

2. **Sign up for Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub

3. **Deploy**
   - Click "New app"
   - Select your repository and branch
   - Set main file: `app.py`
   - Click "Advanced settings"
   - Go to "Secrets" tab
   - Add your environment variables:
     ```
     AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
     AZURE_OPENAI_API_KEY=your-api-key-here
     AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
     AZURE_OPENAI_API_VERSION=2024-02-15-preview
     ```
   - Click "Deploy"

4. **Access your app**
   - Your app will be live at: `https://your-app-name.streamlit.app`
   - Share this URL for judging

## 🚂 Railway (Free Tier Available)

Railway offers a free tier with $5 credit monthly.

### Steps

1. **Sign up**
   - Visit [railway.app](https://railway.app)
   - Sign in with GitHub

2. **Create new project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Configure**
   - Railway auto-detects Python
   - Add environment variables in "Variables" tab:
     - `AZURE_OPENAI_ENDPOINT`
     - `AZURE_OPENAI_API_KEY`
     - `AZURE_OPENAI_DEPLOYMENT`
     - `AZURE_OPENAI_API_VERSION`

4. **Deploy**
   - Railway will auto-deploy
   - Get your public URL from the project dashboard

## 🎨 Render (Free Tier Available)

Render offers free tier for web services.

### Steps

1. **Sign up**
   - Visit [render.com](https://render.com)
   - Sign in with GitHub

2. **Create new Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure**
   - Name: `advisoryai-jarvis`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - Add environment variables:
     - `AZURE_OPENAI_ENDPOINT`
     - `AZURE_OPENAI_API_KEY`
     - `AZURE_OPENAI_DEPLOYMENT`
     - `AZURE_OPENAI_API_VERSION`

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment
   - Get your public URL

## ☁️ Azure App Service

If you're already using Azure OpenAI, deploying to Azure App Service makes sense.

### Steps

1. **Install Azure CLI**
   ```bash
   # macOS
   brew install azure-cli
   
   # Windows
   # Download from azure.microsoft.com
   ```

2. **Login**
   ```bash
   az login
   ```

3. **Create App Service**
   ```bash
   az webapp create \
     --resource-group your-resource-group \
     --plan your-app-service-plan \
     --name your-app-name \
     --runtime "PYTHON:3.9"
   ```

4. **Configure environment variables**
   ```bash
   az webapp config appsettings set \
     --resource-group your-resource-group \
     --name your-app-name \
     --settings \
       AZURE_OPENAI_ENDPOINT="your-endpoint" \
       AZURE_OPENAI_API_KEY="your-key" \
       AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini" \
       AZURE_OPENAI_API_VERSION="2024-02-15-preview"
   ```

5. **Deploy**
   ```bash
   az webapp up \
     --resource-group your-resource-group \
     --name your-app-name \
     --runtime "PYTHON:3.9"
   ```

## 📝 Important Notes

- **Never commit credentials**: Always use environment variables
- **Test locally first**: Ensure app works before deploying
- **Monitor usage**: Keep an eye on API usage and costs
- **Set up alerts**: Configure alerts for errors or high usage
- **Backup data**: For production, ensure data persistence

## 🔍 Verification Checklist

Before submitting, verify:
- [ ] App is publicly accessible (no login required)
- [ ] All environment variables are set correctly
- [ ] Mock data is generated/available
- [ ] App handles basic queries successfully
- [ ] No credentials are exposed in code
- [ ] App doesn't expire immediately after submission

