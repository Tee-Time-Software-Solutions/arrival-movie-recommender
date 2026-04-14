1. Natural language movie search

"Find me a thriller with a twist ending from the 90s"

→ Parse query → filter SQL by genre/year/keywords → return ranked results

2. Taste summary

"What kind of movies have I been liking?"

→ Read liked movie IDs from Redis → aggregate genres/keywords → LLM summarizes the pattern

3. Recommendation explanation

"Why did you recommend Interstellar to me?"
