import os
import time
import asyncio
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import discord
from discord.ext import commands
import chromadb
from google import genai

# --- Mini Web Server για το Render ---
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

# Διαβάζει τα επιλεγμένα κανάλια από τα Environment Variables (χωρισμένα με κόμμα)
raw_channels = os.environ.get("TARGET_CHANNEL_IDS", "")
TARGET_CHANNEL_IDS = [int(ch.strip()) for ch in raw_channels.split(",") if ch.strip()]

client = genai.Client(api_key=GEMINI_API_KEY)

# Custom Embedding Function μέσω Gemini API (μηδενική χρήση RAM στο Render)
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input: list[str]) -> list[list[float]]:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=input
        )
        return [e.values for e in response.embeddings]

embed_fn = GeminiEmbeddingFunction()
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="discord_notes",
    embedding_function=embed_fn
)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

def batch_upsert_worker(docs, metas, ids):
    """Εκτελείται με ασφάλεια σε παρτίδες των 25 για το Gemini API"""
    batch_size = 25
    for i in range(0, len(docs), batch_size):
        success = False
        retries = 5
        while not success and retries > 0:
            try:
                collection.upsert(
                    documents=docs[i:i+batch_size],
                    metadatas=metas[i:i+batch_size],
                    ids=ids[i:i+batch_size]
                )
                success = True
                # Περιμένουμε 15 δευτερόλεπτα για να είμαστε ΑΠΟΛΥΤΑ ασφαλείς κάτω από τα 100 RPM
                time.sleep(15)
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print(f"⚠️ API Limit hit. Waiting 60 seconds... (Batch {i})")
                    time.sleep(60)
                    retries -= 1
                else:
                    print(f"Σφάλμα κατά το batch: {e}")
                    break

async def do_sync(channel_to_notify=None):
    if channel_to_notify:
        await channel_to_notify.send("🔄 Έναρξη συλλογής σημειώσεων από τα επιλεγμένα κανάλια...")

    all_docs = []
    all_metas = []
    all_ids = []

    # Καθορισμός καναλιών: είτε η συγκεκριμένη λίστα είτε όλα εκτός από το chat
    channels_to_scan = []
    if TARGET_CHANNEL_IDS:
        for ch_id in TARGET_CHANNEL_IDS:
            ch = bot.get_channel(ch_id)
            if ch:
                channels_to_scan.append(ch)
    else:
        for guild in bot.guilds:
            for ch in guild.text_channels:
                if ch.id != CHAT_CHANNEL_ID:
                    channels_to_scan.append(ch)

    for channel in channels_to_scan:
        try:
            # limit=None: Διαβάζει ΟΛΑ τα μηνύματα από την αρχή του καναλιού!
            async for msg in channel.history(limit=None):
                if msg.author.bot or not msg.content.strip():
                    continue

                all_docs.append(msg.content)
                all_metas.append({
                    "channel": channel.name,
                    "created_at": msg.created_at.isoformat(),
                    "jump_url": msg.jump_url
                })
                all_ids.append(str(msg.id))
        except Exception as e:
            print(f"Σφάλμα στο κανάλι {channel.name}: {e}")
            continue

    # Φιλτράρισμα: Κρατάμε ΜΟΝΟ τις σημειώσεις που ΔΕΝ υπάρχουν ήδη στο ευρετήριο!
    existing_data = collection.get(include=[])
    existing_ids = set(existing_data['ids']) if existing_data and 'ids' in existing_data else set()
    
    new_docs = []
    new_metas = []
    new_ids = []
    
    for doc, meta, i in zip(all_docs, all_metas, all_ids):
        if i not in existing_ids:
            new_docs.append(doc)
            new_metas.append(meta)
            new_ids.append(i)

    if not new_docs:
        if channel_to_notify:
            await channel_to_notify.send("✅ Δεν υπάρχουν νέες σημειώσεις. Το ευρετήριο είναι ήδη πλήρως ενημερωμένο!")
        return

    if channel_to_notify:
        await channel_to_notify.send(f"⏳ Βρέθηκαν {len(new_docs)} ΝΕΕΣ σημειώσεις. Προσθήκη στο ευρετήριο...")

    await asyncio.to_thread(batch_upsert_worker, new_docs, new_metas, new_ids)

    if channel_to_notify:
        await channel_to_notify.send(f"✅ Ολοκληρώθηκε! Προστέθηκαν επιτυχώς {len(new_docs)} νέες σημειώσεις.")

@bot.event
async def on_ready():
    print(f"Συνδέθηκε ως: {bot.user}")

@bot.command()
async def sync(ctx):
    if ctx.channel.id == CHAT_CHANNEL_ID:
        await do_sync(ctx.channel)

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # Απάντηση ΜΟΝΟ στο chat κανάλι και όχι σε εντολές
    if message.channel.id == CHAT_CHANNEL_ID and not message.content.startswith("!"):
        async with message.channel.typing():
            try:
                results = await asyncio.to_thread(
                    collection.query,
                    query_texts=[message.content],
                    n_results=7
                )

                retrieved = results['documents'][0] if results['documents'] else []
                metas = results['metadatas'][0] if results['metadatas'] else []

                context_blocks = []
                for doc, meta in zip(retrieved, metas):
                    context_blocks.append(f"[{meta['channel']}] {doc}")
                context = "\n---\n".join(context_blocks)

                prompt = (
                    f"Σημειώσεις από τα σημαντικά κανάλια του server μου:\n{context}\n\n"
                    f"Ερώτηση/Σκέψη: {message.content}"
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=dict(
                        system_instruction=(
                            "Είσαι ο προσωπικός μου βοηθός σκέψης. Έχεις πρόσβαση στις προσωπικές μου σημειώσεις "
                            "από τα επιλεγμένα κανάλια του Discord server μου (δίπλα σε κάθε σημείωση αναγράφεται το κανάλι [όνομα]). "
                            "Απάντησε, σύγκρινε, συνδύασε ιδέες και ανάφερε από ποια κανάλια αντλείς πληροφορίες."
                        )
                    )
                )

                reply = response.text
                for i in range(0, len(reply), 1900):
                    await message.reply(reply[i:i+1900])

            except Exception as e:
                await message.reply(f"⚠️ Παρουσιάστηκε σφάλμα: {e}")

    await bot.process_commands(message)

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)