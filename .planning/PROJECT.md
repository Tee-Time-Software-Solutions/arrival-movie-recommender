# Movie Recommender

A full-stack movie recommendation application using collaborative filtering (ALS) with an online learning component for real-time user preference adaptation.

## Stack
- **Backend:** Python 3.13, FastAPI, implicit (ALS), numpy, pandas, scipy, Redis
- **Frontend:** React + TypeScript + Vite
- **ML Pipeline:** Offline ALS training → Online vector updates via swipe feedback
- **External APIs:** TMDB for movie metadata hydration

## Architecture
- Offline pipeline: preprocess → filter → split → build matrix → train ALS → evaluate
- Online serving: Recommender loads artifacts, scores movies via dot product, updates user vectors on swipe
- Feed system: FeedManager manages Redis queue, MovieHydrator fetches TMDB metadata
