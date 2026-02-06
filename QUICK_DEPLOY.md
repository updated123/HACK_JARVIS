# 🚀 Quick Deployment Guide

## Backend on Render (5 minutes)

1. **Go to [render.com](https://render.com)** and sign up/login
2. **Click "New +" → "Web Service"**
3. **Connect GitHub**: Select `updated123/HACK_JARVIS`
4. **Configure**:
   - **Name**: `jarvis-backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend_api:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
5. **Add Environment Variables** (in "Environment" tab):
   ```
   AZURE_OPENAI_ENDPOINT=https://aakashopenai.openai.azure.com/
   AZURE_OPENAI_API_KEY=your-actual-key-here
   AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   ```
6. **Click "Create Web Service"**
7. **Wait 5-10 minutes** for deployment
8. **Copy your backend URL**: `https://jarvis-backend.onrender.com`

## Frontend on Vercel (3 minutes)

1. **Go to [vercel.com](https://vercel.com)** and sign up/login
2. **Click "Add New" → "Project"**
3. **Import from GitHub**: Select `updated123/HACK_JARVIS`
4. **Configure**:
   - **Framework Preset**: Next.js (auto-detected)
   - **Root Directory**: `frontend` ⚠️ **IMPORTANT**
   - **Build Command**: `npm run build` (auto-detected)
5. **Add Environment Variable**:
   - Key: `NEXT_PUBLIC_API_URL`
   - Value: `https://jarvis-backend.onrender.com` (your Render URL from step 8)
6. **Click "Deploy"**
7. **Wait 2-3 minutes** for deployment
8. **Your app is live!** 🎉

## Test Your Deployment

### Test Backend
```bash
curl https://jarvis-backend.onrender.com/health
```

### Test Frontend
Visit your Vercel URL and try:
- Sending a chat message
- Loading the daily briefing

## Troubleshooting

### Backend not starting?
- Check Render logs: Dashboard → Your Service → Logs
- Verify all environment variables are set
- Make sure `backend_api.py` is in the root directory

### Frontend can't connect to backend?
- Check that `NEXT_PUBLIC_API_URL` is set correctly in Vercel
- Check browser console for CORS errors
- Verify backend is running: `curl https://jarvis-backend.onrender.com/health`

### CORS errors?
Update `backend_api.py` line 18:
```python
allow_origins=["https://your-app.vercel.app"]  # Replace with your Vercel URL
```

## URLs for Submission

- **Backend**: `https://jarvis-backend.onrender.com`
- **Frontend**: `https://your-app.vercel.app`

---

**Need help?** Check `DEPLOYMENT_SPLIT.md` for detailed instructions.

