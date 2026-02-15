from .health import router as health_router
from .interactions import router as interactions_router
from .movies import router as movies_router
from .users import router as users_router

routers = [health_router, interactions_router, movies_router, users_router]
