import os
import asyncio
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import discord
from discord.ext import commands
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

# Διαβάζει τα επιλεγμένα κανάλια από τα Environment Variables
raw_channels = os.environ.get("TARGET_CHANNEL_IDS", "")
TARGET_CHANNEL_IDS = [int(ch.strip()) for ch in raw_channels.split(",") if ch.strip()]

client = genai.Client(api_key=GEMINI_API_KEY)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# --- IN-MEMORY MEMORY ---
MEMORY_NOTES = []

async def build_memory():
    """Διαβάζει τα τελευταία μηνύματα και χτίζει τη μνήμη κατά την εκκίνηση"""
    global MEMORY_NOTES
    MEMORY_NOTES.clear()
    
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
                    
    print(f"🔄 Έναρξη φόρτωσης μνήμης από {len(channels_to_scan)} κανάλια...")
    
    for channel in channels_to_scan:
        try:
            # Διαβάζει τα τελευταία 2000 μηνύματα (γρήγορο, δεν κοστίζει τίποτα σε όρια API)
            async for msg in channel.history(limit=2000):
                if msg.author.bot or not msg.content.strip():
                    continue
                MEMORY_NOTES.append(f"[{channel.name}] ({msg.created_at.strftime('%Y-%m-%d %H:%M')}): {msg.content}")
        except Exception as e:
            print(f"Σφάλμα στο κανάλι {channel.name}: {e}")
            continue

    print(f"✅ Η μνήμη φορτώθηκε! Σύνολο σημειώσεων: {len(MEMORY_NOTES)}")

@bot.event
async def on_ready():
    print(f"Συνδέθηκε ως: {bot.user}")
    bot.loop.create_task(build_memory())

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # Αν το μήνυμα γράφτηκε σε ένα από τα κανάλια-στόχους (όχι στο chat), το προσθέτουμε δυναμικά στη μνήμη
    if message.channel.id != CHAT_CHANNEL_ID:
        is_target = False
        if TARGET_CHANNEL_IDS:
            if message.channel.id in TARGET_CHANNEL_IDS:
                is_target = True
        else:
            is_target = True 
            
        if is_target and message.content.strip():
            MEMORY_NOTES.append(f"[{message.channel.name}] ({message.created_at.strftime('%Y-%m-%d %H:%M')}): {message.content}")

    # Απάντηση ΜΟΝΟ στο chat κανάλι (χωρίς prefix πλέον)
    if message.channel.id == CHAT_CHANNEL_ID and not message.content.startswith("!"):
        async with message.channel.typing():
            try:
                # Παίρνουμε τις τελευταίες 3000 σημειώσεις (για απόλυτη ασφάλεια στη μνήμη RAM)
                context_notes = MEMORY_NOTES[-3000:]
                context_str = "\n---\n".join(context_notes)
                
                prompt = (
                    f"Ακολουθούν ΟΛΕΣ οι προσωπικές μου σημειώσεις από τον server μου:\n\n{context_str}\n\n"
                    f"Ερώτηση/Σκέψη του χρήστη: {message.content}"
                )

                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-3.8-flash",
                    contents=prompt,
                    config=dict(
                        system_instruction=(
                            "Είσαι ο προσωπικός μου βοηθός σκέψης. Σου έχω δώσει παραπάνω ΟΛΕΣ τις σημειώσεις μου. "
                            "Απάντησε στην Ερώτηση/Σκέψη μου, συνδυάζοντας ιδέες από τις σημειώσεις μου αν χρειάζεται. "
                            "Ανάφερε σε παρένθεση από ποια κανάλια/ημερομηνίες αντλείς πληροφορίες."
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