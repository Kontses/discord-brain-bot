import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import discord
from discord.ext import commands
import chromadb
from google import genai

# --- Mini Web Server για να ικανοποιεί τα Health Checks του Render ---


class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(b"Discord Bot is running 24/7!")


def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), SimpleHandler)
    server.serve_forever()


# Εκκίνηση του web server σε background thread
threading.Thread(target=run_web_server, daemon=True).start()

# --- Αρχικοποίηση AI Client & Vector DB ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")

client = genai.Client(api_key=GEMINI_API_KEY)
# In-memory client κατάλληλο για cloud container
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="discord_notes")

# --- Discord Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# IDs καναλιών (θα διαβάζονται από τις μεταβλητές περιβάλλοντος)
CHAT_CHANNEL_ID = int(os.environ.get("CHAT_CHANNEL_ID", 0))
NOTES_CHANNELS = [int(ch.strip()) for ch in os.environ.get(
    "NOTES_CHANNEL_IDS", "").split(",") if ch.strip()]


async def do_sync(channel_to_notify=None):
    """Συνάρτηση ευρετηρίασης σημειώσεων."""
    if channel_to_notify:
        await channel_to_notify.send("🔄 Έναρξη συγχρονισμού των σημειώσεων...")

    total_indexed = 0
    for ch_id in NOTES_CHANNELS:
        channel = bot.get_channel(ch_id)
        if not channel:
            continue

        async for msg in channel.history(limit=2000):
            if msg.author.bot or not msg.content.strip():
                continue

            collection.upsert(
                documents=[msg.content],
                metadatas=[{
                    "created_at": msg.created_at.isoformat(),
                    "jump_url": msg.jump_url
                }],
                ids=[str(msg.id)]
            )
            total_indexed += 1

    if channel_to_notify:
        await channel_to_notify.send(
            f"✅ Ολοκληρώθηκε! Διαβάστηκαν {total_indexed} καταχωρήσεις."
        )
    print(f"Συγχρονίστηκαν {total_indexed} μηνύματα.")


@bot.event
async def on_ready():
    print(f"Συνδέθηκε ως: {bot.user}")
    # Αυτόματος συγχρονισμός με το που ανοίγει το bot
    await do_sync()


@bot.command()
async def sync(ctx):
    """Εντολή χειροκίνητου συγχρονισμού"""
    if ctx.channel.id == CHAT_CHANNEL_ID:
        await do_sync(ctx.channel)


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # Αν το μήνυμα είναι στο chat κανάλι και δεν είναι εντολή
    if (message.channel.id == CHAT_CHANNEL_ID and
            not message.content.startswith("!")):
        async with message.channel.typing():
            # 1. Σημασιολογική αναζήτηση στη βάση
            results = collection.query(
                query_texts=[message.content],
                n_results=7
            )

            retrieved = results['documents'][0] if results['documents'] else []
            context = "\n---\n".join(retrieved)

            prompt = (
                "Παρακάτω βρίσκονται σχετικές προσωπικές μου σημειώσεις:\n"
                f"{context}\n\n"
                f"Ερώτηση/Σκέψη: {message.content}"
            )

            # 2. Κλήση Gemini Flash
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=dict(
                    system_instruction=(
                        "Είσαι ο προσωπικός μου βοηθός ανάλυσης. "
                        "Χρησιμοποίησε τις προσωπικές μου σημειώσεις "
                        "από το Context για να απαντήσεις, "
                        "να συγκρίνεις ή να μου υπενθυμίσεις σκέψεις μου. "
                        "Αν δεν προκύπτει απάντηση από τις σημειώσεις, "
                        "ανάφερέ το άμεσα."
                    )
                )
            )

            reply = response.text
            # Χωρισμός σε κομμάτια αν υπερβαίνει το όριο των 2000 χαρακτήρων
            for i in range(0, len(reply), 1900):
                await message.reply(reply[i:i+1900])

    await bot.process_commands(message)

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
