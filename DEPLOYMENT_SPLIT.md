# Split Deployment Guide: Backend (Render) + Frontend (Vercel)

This guide explains how to deploy Jarvis with a split architecture:
- **Backend API**: FastAPI on Render
- **Frontend**: Next.js on Vercel

## 🏗️ Architecture

```
┌─────────────┐         ┌──────────────┐
│   Vercel    │  HTTP   │    Render    │
│  (Frontend) │ ───────> │   (Backend)  │
│  Next.js    │         │   FastAPI    │
└─────────────┘         └──────────────┘
                              │
                              ▼
                        Azure OpenAI API
```

## 🚀 Step 1: Deploy Backend to Render

### 1.1 Prepare Backend

The backend is in `backend_api.py`. It's ready to deploy.

### 1.2 Deploy to Render

1. **Sign up/Login** to [render.com](https://render.com)
2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `updated123/HACK_JARVIS`
   - Configure:
     - **Name**: `jarvis-backend`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn backend_api:app --host 0.0.0.0 --port $PORT`
     - **Plan**: Free (or paid for better performance)

3. **Add Environment Variables**:
   - Go to "Environment" tab
   - Add these variables:
     ```
     AZURE_OPENAI_ENDPOINT=https://aakashopenai.openai.azure.com/
     AZURE_OPENAI_API_KEY=your-actual-api-key-here
     AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
     AZURE_OPENAI_API_VERSION=2024-02-15-preview
     ```

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)
   - Your backend will be at: `https://jarvis-backend.onrender.com`

### 1.3 Test Backend

```bash
# Health check
curl https://jarvis-backend.onrender.com/health

# Test chat
curl -X POST https://jarvis-backend.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What needs my attention today?"}'
```

## 🎨 Step 2: Deploy Frontend to Vercel

### 2.1 Prepare Frontend

The frontend is in the `frontend/` directory (Next.js).

### 2.2 Deploy to Vercel

1. **Sign up/Login** to [vercel.com](https://vercel.com)
2. **Import Project**
   - Click "Add New" → "Project"
   - Import from GitHub: `updated123/HACK_JARVIS`
   - Configure:
     - **Framework Preset**: Next.js
     - **Root Directory**: `frontend`
     - **Build Command**: `npm run build` (auto-detected)
     - **Output Directory**: `.next` (auto-detected)

3. **Add Environment Variables**:
   - Go to "Environment Variables"
   - Add:
     ```
     NEXT_PUBLIC_API_URL=https://jarvis-backend.onrender.com
     ```
   - Replace with your actual Render backend URL

4. **Deploy**
   - Click "Deploy"
   - Wait for deployment (2-3 minutes)
   - Your frontend will be at: `https://your-app.vercel.app`

## 📝 Alternative: Keep Streamlit Frontend

If you prefer to keep Streamlit, you can:
- Deploy backend to Render (as above)
- Update `app.py` to call the backend API instead of initializing JarvisAgent directly
- Deploy Streamlit to Streamlit Cloud (as before)

## 🔧 Configuration Files

### Backend (Render)
- `backend_api.py` - FastAPI application
- `render.yaml` - Render configuration (optional)
- `requirements.txt` - Python dependencies

### Frontend (Vercel)
- `frontend/package.json` - Node.js dependencies
- `frontend/pages/index.tsx` - Main React page
- `frontend/next.config.js` - Next.js configuration
- `frontend/.env.local.example` - Environment variables template

## 🔒 Security Notes

1. **CORS**: Backend allows all origins (`*`). In production, restrict to your Vercel domain:
   ```python
   allow_origins=["https://your-app.vercel.app"]
   ```

2. **API Keys**: Never commit API keys. Use environment variables in both Render and Vercel.

3. **Rate Limiting**: Consider adding rate limiting to the backend API.

## 🧪 Testing

### Test Backend Locally
```bash
cd /path/to/HACK
uvicorn backend_api:app --reload
# Backend runs on http://localhost:8000
```

### Test Frontend Locally
```bash
cd frontend
npm install
npm run dev
# Frontend runs on http://localhost:3000
```

## 📊 API Endpoints

### Backend API (Render)

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /api/chat` - Chat with Jarvis
  ```json
  {
    "message": "What needs my attention today?"
  }
  ```
- `GET /api/briefing` - Get daily briefing
- `POST /api/search?query=...` - Search clients

## 🐛 Troubleshooting

### Backend Issues
- Check Render logs: Dashboard → Your Service → Logs
- Verify environment variables are set
- Check that `backend_api.py` is in the root directory

### Frontend Issues
- Check Vercel logs: Dashboard → Your Project → Functions → Logs
- Verify `NEXT_PUBLIC_API_URL` is set correctly
- Check browser console for CORS errors

### CORS Errors
If you see CORS errors, update `backend_api.py`:
```python
allow_origins=["https://your-app.vercel.app"]
```

## 🎯 Benefits of Split Architecture

1. **Scalability**: Backend and frontend scale independently
2. **Performance**: Frontend can be cached on CDN (Vercel)
3. **Flexibility**: Easy to swap frontend framework
4. **Cost**: Free tiers on both Render and Vercel
5. **Separation**: Clear separation of concerns

## 📚 Next Steps

1. Deploy backend to Render
2. Deploy frontend to Vercel
3. Update CORS settings in backend
4. Test the full stack
5. Share both URLs for submission

---

**Backend URL**: `https://jarvis-backend.onrender.com`  
**Frontend URL**: `https://your-app.vercel.app`

