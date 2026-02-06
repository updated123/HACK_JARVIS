# Jarvis Frontend

Next.js frontend for Jarvis AI Assistant.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Set environment variables:
```bash
cp .env.local.example .env.local
# Edit .env.local and set NEXT_PUBLIC_API_URL to your Render backend URL
```

3. Run development server:
```bash
npm run dev
```

4. Build for production:
```bash
npm run build
npm start
```

## Deploy to Vercel

1. Connect your GitHub repository to Vercel
2. Set root directory to `frontend`
3. Add environment variable: `NEXT_PUBLIC_API_URL=https://your-backend.onrender.com`
4. Deploy!

