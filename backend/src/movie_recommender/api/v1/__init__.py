from .chatbot import router as chatbot_router
from .health import router as health_router
from .interactions import router as interactions_router
from .movies import router as movies_router
from .onboarding import router as onboarding_router
from .users import router as users_router
from .watchlist import router as watchlist_router

routers = [
    chatbot_router,
    health_router,
    interactions_router,
    movies_router,
    onboarding_router,
    users_router,
    watchlist_router,
]
