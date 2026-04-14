
## Functionality
Note: the chatbot will use as tool different function that create CRUD operations on the database. Make sure this CRUD opeartions are READ-only (i.e., the chatbot cant alter erroneously something). The tools are what allow the LLM to have the prper context to respond with. Without them, they'd be guessing.


1. Natural language movie search

"Find me a thriller with a twist ending from the 90s"

→ Parse query → filter SQL by genre/year/keywords → return ranked results

2. Taste summary

"What kind of movies have I been liking?"

→ Read liked movie IDs from Redis → aggregate genres/keywords → LLM summarizes the pattern

## Working tree
### Backend
- Write the CRUD operations inside /database/CRUD
- Write the core logic of the chatbot in services/chatbot (to be created)
- Create a new endpoint inside api/chatbot. This is what the frontend should call. Use SSE events + langraph streaming API to provide a real time rendering experience.
- Frontend should also be updated


## Tech stack
- OpenRouter for the llm provider. We do this in order to be able to claim we can access free open source models
- LAnggraph + langchain for the agent

## Debuggin
1) Make sure u have run the pipeline (make -C backend recommender-train-als)
2) Make sure u have `backend/env_config/synced/.env.dev` populated as per indicated by .env.dev.example
3) Run `make dev-rebuild` (first time or when dependencies change) or `make dev-start` (when dependencies havent changed this restarts the app faster) to have the app running (frontend + backend)


## Results, Expectations
A video demo proving that after onboarding + some swipes, when going into the chatbot section in the frotnend if u ask it question such as "taste summary" the response makes sense based on prior actions. Another question to be tested against: `"Find me a thriller with a twist ending from the 90s"`this should check inside the db for the info via the tools and respond with a movie suggestion. When you click on the movie (ideally add this, but not mandatory if time is running short) you have the same UI element (already coded) taht gets opened when you swipe down a recomemndation (i.e., metadata feed.)
