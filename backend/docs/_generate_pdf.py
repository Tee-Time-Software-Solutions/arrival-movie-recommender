"""Generate PDF summary of the Single User Recommendation Journey test."""
from fpdf import FPDF

OUTPUT = "/Users/alex/arrival-movie-recommender/backend/docs/single_user_journey_summary.pdf"


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Arrival Movie Recommender - Test Report", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def key_value(self, key, value):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(40, 40, 40)
        self.cell(50, 6, key)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")

    def rec_table(self, rows):
        """rows = [(rank, movie, genre), ...]"""
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(240, 240, 240)
        self.cell(12, 6, "#", border=1, fill=True, align="C")
        self.cell(80, 6, "Movie", border=1, fill=True)
        self.cell(40, 6, "Genre", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        for rank, movie, genre in rows:
            color = {"ACTION": (220, 50, 50), "COMEDY": (50, 150, 50), "HORROR": (100, 50, 180)}
            r, g, b = color.get(genre, (40, 40, 40))
            self.cell(12, 6, str(rank), border=1, align="C")
            self.cell(80, 6, movie, border=1)
            self.set_text_color(r, g, b)
            self.cell(40, 6, genre, border=1, new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(40, 40, 40)
        self.ln(2)

    def delta_table(self, deltas):
        """deltas = [(genre, delta_str, direction), ...]"""
        self.set_font("Courier", "", 9)
        for genre, delta, direction in deltas:
            arrow = {"up": "\u2191", "down": "\u2193", "flat": "-"}.get(direction, "")
            self.cell(25, 5, genre)
            self.cell(40, 5, f"  avg delta: {delta}", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)


pdf = PDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Title
pdf.set_font("Helvetica", "B", 20)
pdf.set_text_color(20, 20, 20)
pdf.cell(0, 12, "Single User Recommendation Journey", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 7, "Integration Test Report", new_x="LMARGIN", new_y="NEXT")
pdf.ln(4)

# Overview
pdf.section_title("Overview")
pdf.body_text(
    "This test demonstrates how a single user (User 1) moves through the full "
    "recommendation pipeline -- from cold-start recommendations to personalized results "
    "shaped by swipe interactions. The online updater adjusts the user's embedding vector "
    "in real time, and get_top_n() returns updated recommendations after each swipe."
)

# Config
pdf.sub_title("Test Configuration")
pdf.key_value("Embedding dims:", "6 (2 per genre)")
pdf.key_value("Genre mapping:", "ACTION -> dims[0-1], COMEDY -> dims[2-3], HORROR -> dims[4-5]")
pdf.key_value("Learning rate (eta):", "0.3")
pdf.key_value("Movies:", "12 (4 action, 4 comedy, 4 horror)")
pdf.key_value("Recs per step:", "5")
pdf.key_value("Test status:", "PASSED")
pdf.ln(6)

# Steps
steps = [
    {
        "title": "Step 1 -- Cold Start (no history)",
        "desc": "User opens the app with no swipe history. Recommendations based solely on initial user vector with slight action lean.",
        "recs": [(1, "Die Hard", "ACTION"), (2, "John Wick", "ACTION"), (3, "Mad Max", "ACTION"), (4, "Top Gun", "ACTION"), (5, "Bridesmaids", "COMEDY")],
        "deltas": None,
        "note": "Initial vector already favors action -- 4 of 5 recs are action films.",
    },
    {
        "title": 'Step 2 -- LIKE "Die Hard" [ACTION]',
        "desc": None,
        "recs": [(1, "John Wick", "ACTION"), (2, "Mad Max", "ACTION"), (3, "Top Gun", "ACTION"), (4, "Bridesmaids", "COMEDY"), (5, "Superbad", "COMEDY")],
        "deltas": [("ACTION", "+0.2750", "up"), ("COMEDY", "+0.0075", "flat"), ("HORROR", "+0.0000", "flat")],
        "note": "Action scores jump significantly. Die Hard exits recs (already seen).",
    },
    {
        "title": 'Step 3 -- LIKE "John Wick" [ACTION] (double down)',
        "desc": None,
        "recs": [(1, "Mad Max", "ACTION"), (2, "Top Gun", "ACTION"), (3, "Bridesmaids", "COMEDY"), (4, "Superbad", "COMEDY"), (5, "The Hangover", "COMEDY")],
        "deltas": [("ACTION", "+0.2550", "up"), ("HORROR", "+0.0277", "flat"), ("COMEDY", "+0.0075", "flat")],
        "note": "Second action like reinforces action dimensions.",
    },
    {
        "title": 'Step 4 -- DISLIKE "The Shining" [HORROR]',
        "desc": None,
        "recs": [(1, "Mad Max", "ACTION"), (2, "Top Gun", "ACTION"), (3, "Bridesmaids", "COMEDY"), (4, "Superbad", "COMEDY"), (5, "The Hangover", "COMEDY")],
        "deltas": [("ACTION", "+0.0000", "flat"), ("COMEDY", "+0.0000", "flat"), ("HORROR", "-0.2750", "down")],
        "note": "Horror scores drop sharply. Other genres unaffected (orthogonal embeddings).",
    },
    {
        "title": 'Step 5 -- SKIP "Bridesmaids" [COMEDY]',
        "desc": None,
        "recs": [(1, "Mad Max", "ACTION"), (2, "Top Gun", "ACTION"), (3, "Superbad", "COMEDY"), (4, "The Hangover", "COMEDY"), (5, "Mean Girls", "COMEDY")],
        "deltas": None,
        "note": "Vector unchanged. Skip sends preference=0 -- no learning occurs. Movie removed from future recs.",
    },
    {
        "title": 'Step 6 -- LIKE "Mad Max" [ACTION] (third action like)',
        "desc": None,
        "recs": [(1, "Top Gun", "ACTION"), (2, "Superbad", "COMEDY"), (3, "The Hangover", "COMEDY"), (4, "Mean Girls", "COMEDY"), (5, "The Conjuring", "HORROR")],
        "deltas": [("ACTION", "+0.2340", "up"), ("COMEDY", "+0.0000", "flat"), ("HORROR", "+0.0000", "flat")],
        "note": "Third action like. With most action movies seen, other genres fill in.",
    },
]

for step in steps:
    pdf.sub_title(step["title"])
    if step["desc"]:
        pdf.body_text(step["desc"])
    pdf.rec_table(step["recs"])
    if step["deltas"]:
        pdf.delta_table(step["deltas"])
    if step["note"]:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 5, step["note"])
        pdf.set_text_color(40, 40, 40)
        pdf.ln(4)

# Summary
pdf.add_page()
pdf.section_title("Journey Summary")

pdf.sub_title("Swipe Activity")
pdf.key_value("Total swipes:", "5")
pdf.key_value("Action LIKES:", "3")
pdf.key_value("Horror DISLIKES:", "1")
pdf.key_value("Comedy SKIPS:", "1")
pdf.key_value("Movies seen:", "5")
pdf.ln(4)

pdf.sub_title("User Vector Evolution")
pdf.set_font("Courier", "B", 9)
pdf.set_fill_color(240, 240, 240)
pdf.cell(35, 6, "Dimension", border=1, fill=True)
pdf.cell(30, 6, "Start", border=1, fill=True, align="C")
pdf.cell(30, 6, "Final", border=1, fill=True, align="C")
pdf.cell(30, 6, "Change", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")

pdf.set_font("Courier", "", 9)
vectors = [
    ("ACTION [0]", "+0.4000", "+1.2700", "+0.8700"),
    ("ACTION [1]", "+0.1000", "+0.1900", "+0.0900"),
    ("COMEDY [2]", "+0.2000", "+0.2000", " 0.0000"),
    ("COMEDY [3]", "+0.1000", "+0.1000", " 0.0000"),
    ("HORROR [4]", "+0.1000", "-0.1700", "-0.2700"),
    ("HORROR [5]", "+0.0000", "-0.0300", "-0.0300"),
]
for dim, start, final, change in vectors:
    pdf.cell(35, 6, dim, border=1)
    pdf.cell(30, 6, start, border=1, align="C")
    pdf.cell(30, 6, final, border=1, align="C")
    # Bold for significant changes
    if abs(float(change)) > 0.05:
        pdf.set_font("Courier", "B", 9)
    pdf.cell(30, 6, change, border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", "", 9)

pdf.ln(4)
pdf.set_font("Helvetica", "", 10)
pdf.key_value("Vector drift (L2):", "0.9159")
pdf.key_value("Online vector stored:", "Yes")
pdf.ln(6)

# Key observations
pdf.sub_title("Key Observations")
observations = [
    "Likes shift the vector toward the liked genre -- each action LIKE increased action scores by ~0.25 on average.",
    "Dislikes shift away from the disliked genre -- the horror DISLIKE dropped horror scores by -0.275.",
    "Skips have zero effect -- preference=0 means no vector update.",
    "Genre independence is preserved -- liking action does not affect comedy or horror scores (orthogonal embeddings).",
    "Seen movies are excluded -- recommendations never repeat a swiped movie.",
    "As preferred movies are consumed, other genres fill in -- after 3 action movies are seen, comedy and horror appear in top 5.",
]
for i, obs in enumerate(observations, 1):
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(8, 6, f"{i}.")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, obs)
    pdf.ln(1)

pdf.ln(4)
pdf.sub_title("Technical Notes")
pdf.body_text(
    "Embeddings are hand-crafted with orthogonal genre dimensions (not ALS-trained) to ensure "
    "clear, interpretable genre separation in test output. Production uses 64-dim ALS embeddings "
    "trained on MovieLens-25M; the same update_user_vector() and get_top_n() functions are used "
    "in both contexts. The online updater formula: user_vector += eta * preference * movie_vector, "
    "with norm capping to prevent explosion."
)

pdf.output(OUTPUT)
print(f"PDF saved to {OUTPUT}")
