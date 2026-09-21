import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import discord
from discord.ext import commands
import chromadb
from google import genai

# --- Mini Web Server για να κρατάει το Render ενεργό ---
class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(b"Discord Bot is alive!")

def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), SimpleHandler)
    server.serve_forever()

threading.Thread(target=run_web_server, daemon=True).start()

# --- Clients Setup ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
CHAT_CHANNEL_ID = int(os.environ.get("CHAT_CHANNEL_ID", 0))

client = genai.Client(api_key=GEMINI_API_KEY)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="discord_notes")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

async def do_sync(channel_to_notify=None):
    """Διαβάζει αυτόματα όλα τα κανάλια κειμένου εκτός από το chat κανάλι."""
    if channel_to_notify:
        await channel_to_notify.send("🔄 Έναρξη συγχρονισμού όλων των καναλιών του server...")

    total_indexed = 0
    for guild in bot.guilds:
        for channel in guild.text_channels:
            # Εξαιρούμε μόνο το κανάλι όπου μιλάμε με το bot
            if channel.id == CHAT_CHANNEL_ID:
                continue

            try:
                # Διαβάζει έως 1500 μηνύματα από κάθε κανάλι σημειώσεων
                async for msg in channel.history(limit=1500):
                    if msg.author.bot or not msg.content.strip():
                        continue

                    collection.upsert(
                        documents=[msg.content],
                        metadatas=[{
                            "channel": channel.name,
                            "created_at": msg.created_at.isoformat(),
                            "jump_url": msg.jump_url
                        }],
                        ids=[str(msg.id)]
                    )
                    total_indexed += 1
            except discord.Forbidden:
                continue

    if channel_to_notify:
        await channel_to_notify.send(f"✅ Συγχρονίστηκαν επιτυχώς {total_indexed} σημειώσεις από όλα τα κανάλια!")
    print(f"Συνολικά καταχωρήθηκαν {total_indexed} μηνύματα.")

@bot.event
async def on_ready():
    print(f"Συνδέθηκε ως: {bot.user}")
    await do_sync()

@bot.command()
async def sync(ctx):
    """Χειροκίνητος συγχρονισμός"""
    if ctx.channel.id == CHAT_CHANNEL_ID:
        await do_sync(ctx.channel)

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # Απάντηση μόνο στο dedicated κανάλι συζήτησης
    if message.channel.id == CHAT_CHANNEL_ID and not message.content.startswith("!"):
        async with message.channel.typing():
            # 1. Σημασιολογική αναζήτηση στη βάση
            results = collection.query(
                query_texts=[message.content],
                n_results=8
            )

            retrieved = results['documents'][0] if results['documents'] else []
            metas = results['metadatas'][0] if results['metadatas'] else []

            # Σύνθεση του context μαζί με το όνομα του αντίστοιχου καναλιού
            context_blocks = []
            for doc, meta in zip(retrieved, metas):
                context_blocks.append(f"[{meta['channel']}] {doc}")
            context = "\n---\n".join(context_blocks)

            prompt = (
                f"Σημειώσεις από διάφορα κανάλια του server μου:\n{context}\n\n"
                f"Ερώτηση/Σκέψη: {message.content}"
            )

            # 2. Κλήση Gemini Flash
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=dict(
                    system_instruction=(
                        "Είσαι ο προσωπικός μου βοηθός σκέψης. Έχεις πρόσβαση στις σημειώσεις μου "
                        "από τα κανάλια του Discord server μου (δίπλα σε κάθε σημείωση αναγράφεται το κανάλι [όνομα]). "
                        "Απάντησε, σύγκρινε, συνδύασε ιδέες και ανάφερε από ποια κανάλια αντλείς πληροφορίες."
                    )
                )
            )

            reply = response.text
            # Χωρισμός αν υπερβαίνει το όριο των 2000 χαρακτήρων του Discord
            for i in range(0, len(reply), 1900):
                await message.reply(reply[i:i+1900])

    await bot.process_commands(message)

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)