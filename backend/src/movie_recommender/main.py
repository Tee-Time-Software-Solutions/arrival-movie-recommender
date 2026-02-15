from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
      # Set-up depedendencies
      yield
      # Shut down dependencies

app = FastAPI(title="Movie Recommender",
              lifespan=lifespan)
app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_methods=["*"],
      allow_headers=["*"],
      allow_credentials=True,
)

routers = [

]

for router in routers:
      app.include_router(router, prefix="/api")
