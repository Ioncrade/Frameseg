# FrameSeg frontend

Next.js 16 studio UI for the FrameSeg SAM2 rotoscoping backend.

## Run locally

Start the backend on port 8000, then run:

```bash
npm install
npm run dev
```

Open `http://localhost:3000`. The frontend uses
`http://localhost:8000` by default.

For a different API location, copy `.env.example` to `.env.local` and change:

```dotenv
NEXT_PUBLIC_API_URL=http://localhost:8000
```

The matching frontend origin must also be present in the backend's
`FRONTEND_ORIGINS` setting.

## Workflow

1. Drop an H.264/H.265 MP4 or MOV clip into the studio.
2. Click the subject in the first frame to generate a whole-object mask.
3. Use **Add** and **Subtract** clicks to correct the mask.
4. Accept the mask and monitor the streaming render.
5. Download the transparent ProRes 4444 MOV.

## Checks

```bash
npm run lint
npm run build
```
