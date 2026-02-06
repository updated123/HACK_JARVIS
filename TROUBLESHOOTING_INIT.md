# 🔧 Troubleshooting: "Jarvis agent not initialized"

## Quick Diagnosis

### Step 1: Check Health Endpoint

Visit or curl your backend health endpoint:
```bash
curl https://your-backend.onrender.com/health
```

This will show you:
- ✅ Which environment variables are set/missing
- ✅ Which files exist
- ✅ The exact error message if initialization failed

### Step 2: Check Render Logs

1. Go to Render Dashboard → Your Service → **Logs**
2. Look for initialization messages:
   - `✓ Credentials loaded` - Good!
   - `❌ Configuration Error` - Missing env vars
   - `❌ Import Error` - Missing dependencies
   - `❌ Initialization Error` - Other issue

## Common Issues & Fixes

### Issue 1: Missing Environment Variables

**Symptoms:**
- Health endpoint shows `"AZURE_OPENAI_ENDPOINT": "missing"`
- Logs show: `❌ Configuration Error: Missing Azure OpenAI credentials`

**Fix:**
1. Go to Render Dashboard → Your Service → **Environment**
2. Add these variables:
   ```
   AZURE_OPENAI_ENDPOINT = https://aakashopenai.openai.azure.com/
   AZURE_OPENAI_API_KEY = your-actual-api-key-here
   AZURE_OPENAI_DEPLOYMENT = gpt-4o-mini
   AZURE_OPENAI_API_VERSION = 2024-02-15-preview
   ```
3. **Important**: Endpoint must end with `/`
4. Click **Save Changes**
5. Service will auto-redeploy

### Issue 2: Invalid API Key

**Symptoms:**
- Health endpoint shows credentials as "set"
- But initialization still fails
- Logs show Azure OpenAI authentication errors

**Fix:**
1. Verify your API key is correct in Azure Portal
2. Make sure key hasn't expired
3. Check that the key has proper permissions
4. Update in Render → Environment → `AZURE_OPENAI_API_KEY`
5. Redeploy

### Issue 3: Endpoint Format Wrong

**Symptoms:**
- Endpoint is set but initialization fails
- Azure OpenAI connection errors

**Fix:**
- Endpoint must be: `https://your-resource.openai.azure.com/`
- ✅ Correct: `https://aakashopenai.openai.azure.com/`
- ❌ Wrong: `https://aakashopenai.openai.azure.com` (missing `/`)
- ❌ Wrong: `aakashopenai.openai.azure.com` (missing `https://`)

### Issue 4: Missing Dependencies

**Symptoms:**
- Logs show: `❌ Import Error: No module named '...'`
- Build succeeds but runtime fails

**Fix:**
1. Check `requirements.txt` includes all packages
2. Verify build logs show all packages installed
3. If missing, add to `requirements.txt` and redeploy

### Issue 5: Vector Store Issues

**Symptoms:**
- Initialization fails during vector store setup
- File permission errors

**Fix:**
- This should auto-fix on retry
- Try manual reinitialize: `POST /api/reinitialize`

## Manual Retry

If initialization failed, you can manually retry:

```bash
# Retry initialization
curl -X POST https://your-backend.onrender.com/api/reinitialize
```

Or visit in browser:
```
https://your-backend.onrender.com/api/reinitialize
```

## Step-by-Step Debugging

1. **Check Health Endpoint**
   ```bash
   curl https://your-backend.onrender.com/health
   ```
   - Note what's missing

2. **Check Render Logs**
   - Look for `❌` error messages
   - Copy the exact error

3. **Verify Environment Variables**
   - Go to Render → Environment tab
   - Check all 4 variables are set
   - Verify endpoint ends with `/`

4. **Try Manual Reinitialize**
   ```bash
   curl -X POST https://your-backend.onrender.com/api/reinitialize
   ```

5. **Check Response**
   - Should return: `{"status": "success", ...}`
   - If failed, check the error message

## Expected Log Output (Success)

When working correctly, you should see:
```
============================================================
Starting Jarvis Backend API...
============================================================
✓ Credentials loaded (endpoint: https://aakashopenai.openai..., deployment: gpt-4o-mini)
✓ Mock data file exists
Initializing vector store...
✓ Vector store already exists
✓ Compliance tracker initialized
Initializing JarvisAgent...
✓ JarvisAgent initialized successfully
============================================================
Backend initialized successfully!
============================================================
```

## Expected Health Response (Success)

```json
{
  "status": "healthy",
  "agent_initialized": true,
  "compliance_tracker_initialized": true,
  "environment_variables": {
    "AZURE_OPENAI_ENDPOINT": "set",
    "AZURE_OPENAI_API_KEY": "set",
    "AZURE_OPENAI_DEPLOYMENT": "gpt-4o-mini",
    "AZURE_OPENAI_API_VERSION": "2024-02-15-preview"
  },
  "files": {
    "mock_data.json": "exists",
    "chroma_db": "exists"
  }
}
```

## Still Not Working?

1. **Check Render Logs** - Most errors are logged there
2. **Verify Environment Variables** - All 4 must be set correctly
3. **Try Manual Reinitialize** - `POST /api/reinitialize`
4. **Check Azure OpenAI** - Verify your Azure resource is active
5. **Redeploy** - Sometimes a fresh deploy fixes issues

## Quick Checklist

- [ ] `AZURE_OPENAI_ENDPOINT` is set and ends with `/`
- [ ] `AZURE_OPENAI_API_KEY` is set and valid
- [ ] `AZURE_OPENAI_DEPLOYMENT` is set (default: `gpt-4o-mini`)
- [ ] `AZURE_OPENAI_API_VERSION` is set (default: `2024-02-15-preview`)
- [ ] Health endpoint shows all variables as "set"
- [ ] Render logs show successful initialization
- [ ] Tried manual reinitialize endpoint

---

**Most Common Issue**: Missing or incorrectly formatted `AZURE_OPENAI_ENDPOINT` (must end with `/`)

