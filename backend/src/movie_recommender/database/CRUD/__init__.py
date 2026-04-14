from .interactions import create_swipe as create_swipe
from .watchlist import (
    add_to_watchlist as add_to_watchlist,
    get_user_watchlist as get_user_watchlist,
    remove_from_watchlist as remove_from_watchlist,
)
from .movies import (
    get_movie_by_id as get_movie_by_id,
    movie_to_details as movie_to_details,
    save_hydrated_movie as save_hydrated_movie,
)
from .users import (
    create_user as create_user,
    get_user_analytics as get_user_analytics,
    get_user_by_firebase_uid as get_user_by_firebase_uid,
    get_user_excluded_genres as get_user_excluded_genres,
    get_user_included_genres as get_user_included_genres,
    get_user_preferences as get_user_preferences,
    update_user_preferences as update_user_preferences,
)
