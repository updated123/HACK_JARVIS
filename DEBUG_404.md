# 🔍 Debugging 404 Errors

## Common Causes of 404 Errors

### 1. Wrong URL / Path

**Check your backend URL:**
- Backend should be: `https://your-backend.onrender.com`
- Make sure there's no trailing slash when calling endpoints
- Example: `https://jarvis-backend.onrender.com/api/chat` ✅
- Wrong: `https://jarvis-backend.onrender.com/api/chat/` ❌

### 2. Backend Not Running

**Test if backend is accessible:**
```bash
curl https://your-backend.onrender.com/
```

Should return:
```json
{
  "status": "ok",
  "service": "Jarvis API",
  "agent_ready": true
}
```

### 3. Wrong Endpoint Path

**Available endpoints:**
- `GET /` - Root (service status)
- `GET /health` - Health check
- `GET /api` - List all endpoints
- `GET /docs` - API documentation
- `POST /api/chat` - Chat endpoint
- `GET /api/briefing` - Daily briefing
- `POST /api/search?query=...` - Search clients

### 4. Frontend Environment Variable

**Check Vercel environment variable:**
- Variable name: `NEXT_PUBLIC_API_URL`
- Value should be: `https://your-backend.onrender.com`
- **No trailing slash!**

### 5. CORS Issues (Browser Console)

If you see CORS errors in browser console:
- Backend allows all origins by default
- Check browser console (F12) for specific error

## 🔧 Step-by-Step Debugging

### Step 1: Test Backend Directly

```bash
# Test root endpoint
curl https://your-backend.onrender.com/

# Test health endpoint
curl https://your-backend.onrender.com/health

# Test API info
curl https://your-backend.onrender.com/api
```

### Step 2: Test Chat Endpoint

```bash
curl -X POST https://your-backend.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

### Step 3: Check Frontend

1. **Open browser console** (F12)
2. **Check Network tab** when making a request
3. **Look for:**
   - What URL is being called?
   - What's the response status?
   - Any CORS errors?

### Step 4: Verify Environment Variable

In Vercel:
1. Go to Settings → Environment Variables
2. Check `NEXT_PUBLIC_API_URL` is set
3. Value should be: `https://your-backend.onrender.com` (no trailing slash)

### Step 5: Check Render Logs

1. Go to Render Dashboard
2. Click your service
3. Go to "Logs" tab
4. Look for:
   - Startup messages
   - Any error messages
   - "Backend initialized successfully"

## 🐛 Common 404 Scenarios

### Scenario 1: Frontend shows 404

**Problem:** Frontend can't find the backend

**Solution:**
1. Check `NEXT_PUBLIC_API_URL` in Vercel
2. Make sure backend URL is correct
3. Test backend directly with curl

### Scenario 2: Backend returns 404

**Problem:** Wrong endpoint path

**Solution:**
- Use `/api/chat` not `/chat`
- Use `/api/briefing` not `/briefing`
- Check `/api` endpoint for all available paths

### Scenario 3: Backend not deployed

**Problem:** Service not running on Render

**Solution:**
1. Check Render dashboard
2. Make sure service is "Live"
3. Check logs for errors
4. Verify environment variables are set

## ✅ Quick Test Checklist

- [ ] Backend root (`/`) returns 200 OK
- [ ] Backend health (`/health`) returns 200 OK
- [ ] Backend API info (`/api`) returns 200 OK
- [ ] Frontend `NEXT_PUBLIC_API_URL` is set correctly
- [ ] No trailing slash in backend URL
- [ ] Backend is "Live" on Render
- [ ] No errors in Render logs
- [ ] No CORS errors in browser console

## 📝 Example Working URLs

**Backend (Render):**
```
https://jarvis-backend.onrender.com/
https://jarvis-backend.onrender.com/health
https://jarvis-backend.onrender.com/api
https://jarvis-backend.onrender.com/api/chat
https://jarvis-backend.onrender.com/api/briefing
```

**Frontend (Vercel):**
```
https://jarvis-frontend.vercel.app
```

**Environment Variable in Vercel:**
```
NEXT_PUBLIC_API_URL=https://jarvis-backend.onrender.com
```

## 🆘 Still Getting 404?

1. **Check the exact URL** being called (browser Network tab)
2. **Compare with backend endpoints** (visit `/api` endpoint)
3. **Test backend directly** with curl or Postman
4. **Check Render logs** for startup errors
5. **Verify environment variable** is set in Vercel

---

**Need more help?** Check the `/api` endpoint on your backend to see all available endpoints!

