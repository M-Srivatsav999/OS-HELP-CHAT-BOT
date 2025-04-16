from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import pipeline
import logging
import asyncio
import nest_asyncio  # Import nest_asyncio
from googlesearch import search  # Import Google search functionality
import requests
from bs4 import BeautifulSoup

# Apply nest_asyncio to avoid "event loop already running" error
nest_asyncio.apply()

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Load both models: DistilBERT for Question-Answering and GPT-Neo for Text Generation
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
gpt_neo_model = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# A dictionary to store user profiles (using Telegram ID as key)
user_profiles = {}

# Rich FAQ dataset for context
faq_context = """
Blue Screen Error: This issue usually occurs due to hardware or driver problems.\n 
You can try the following steps to resolve it:\n
1. Note the error code displayed.\n
2. Restart your computer.\n
3. Uninstall recent drivers or updates if the problem started recently.\n

Slow Performance: If your system is slow, try the following:\n
1. Close background programs that you aren't using.\n
2. Run antivirus software to check for malware.\n
3. Consider upgrading your hardware, such as adding more RAM or an SSD.\n

Internet Connectivity Issues: If you're facing connectivity problems:\n
1. Ensure that your network drivers are up to date.\n
2. Restart your modem and router.\n
3. Check if the problem is specific to your device or all devices.\n

Virus Problems: Run a full system scan using reliable antivirus software. Make sure your antivirus definitions are up to date.\n

Software Updates: Ensure both your operating system and applications are up to date to ensure maximum compatibility and performance.\n
"""

# Expanded FAQ data for quick matching
faq_data = {
    # Windows OS Issues
    "blue screen": (
        "To troubleshoot a blue screen error:\n"
        "1. Note any error codes displayed.\n"
        "2. Restart your computer.\n"
        "3. Check for recent hardware or software changes.\n"
        "4. Run 'sfc /scannow' to check for corrupted system files.\n"
        "5. Use the 'Event Viewer' to inspect system logs for errors.\n"
        "6. If the problem persists, consider a clean Windows reinstallation."
    ),
    "slow performance": (
        "To improve performance on Windows:\n"
        "1. Close unnecessary applications and background processes.\n"
        "2. Run a full system virus scan with reliable antivirus software.\n"
        "3. Use 'Task Manager' to identify resource-hogging programs.\n"
        "4. Check for system updates (Windows Update).\n"
        "5. Defragment your hard drive (not applicable for SSDs).\n"
        "6. Increase virtual memory if your RAM is low.\n"
        "7. Consider upgrading your hardware (more RAM, SSD, etc.)."
    ),
    "internet connection": (
        "To troubleshoot internet connectivity issues:\n"
        "1. Check if Wi-Fi is enabled and you're connected to the correct network.\n"
        "2. Restart your modem and router.\n"
        "3. Run the Windows Network Troubleshooter.\n"
        "4. Disable and re-enable the network adapter in 'Device Manager'.\n"
        "5. Check for updated network drivers.\n"
        "6. If using VPN or proxy, try disabling them temporarily."
    ),
    "software update": (
        "For software and Windows updates:\n"
        "1. Open 'Windows Update' from the Control Panel or Settings.\n"
        "2. Ensure automatic updates are enabled.\n"
        "3. Regularly check for updates to critical software (antivirus, drivers, etc.).\n"
        "4. Restart the computer after significant updates."
    ),
    "virus": (
        "If you suspect a virus or malware:\n"
        "1. Run a full antivirus scan with updated definitions.\n"
        "2. Boot into Safe Mode and run a scan again for persistent threats.\n"
        "3. Consider using specialized anti-malware tools (e.g., Malwarebytes).\n"
        "4. Check for suspicious programs in 'Task Manager'.\n"
        "5. Reset your browser settings if you notice pop-ups or unwanted toolbars."
    ),

    # macOS Issues
    "mac slow performance": (
        "To improve performance on macOS:\n"
        "1. Check the Activity Monitor for resource-heavy processes.\n"
        "2. Restart your Mac and close unused apps.\n"
        "3. Clear storage space by removing unnecessary files or moving them to external drives.\n"
        "4. Disable startup programs from System Preferences > Users & Groups.\n"
        "5. Run Disk Utility to repair disk permissions.\n"
        "6. Consider upgrading RAM or switching to an SSD."
    ),
    "mac wifi issues": (
        "If you're having Wi-Fi issues on macOS:\n"
        "1. Restart your router and Mac.\n"
        "2. Go to System Preferences > Network, and check Wi-Fi settings.\n"
        "3. Forget and reconnect to the Wi-Fi network.\n"
        "4. Reset the 'PRAM' and 'SMC' on your Mac.\n"
        "5. Check if other devices are connecting to the same network."
    ),
    "mac not booting": (
        "If your Mac is not booting:\n"
        "1. Reset 'PRAM' by holding 'Option + Command + P + R' during startup.\n"
        "2. Boot into Safe Mode by holding 'Shift' during startup.\n"
        "3. Run 'Disk Utility' from macOS Recovery Mode (Command + R).\n"
        "4. Reinstall macOS if none of the above steps work."
    ),
    "mac software update": (
        "To update macOS or applications:\n"
        "1. Go to System Preferences > Software Update.\n"
        "2. Check if your Mac is compatible with the latest macOS version.\n"
        "3. For App Store apps, open the App Store and check for updates.\n"
        "4. Ensure enough disk space for the update (at least 10-15 GB)."
    ),

    # Linux Issues
    "linux slow performance": (
        "To improve performance on Linux:\n"
        "1. Check resource usage with 'top' or 'htop' command.\n"
        "2. Disable unnecessary services from starting up with 'systemctl'.\n"
        "3. Clear temporary files and logs with 'sudo apt-get clean' or 'sudo dnf clean all'.\n"
        "4. Use a lightweight desktop environment (LXDE, XFCE).\n"
        "5. Add more swap space if your RAM is limited.\n"
        "6. Upgrade hardware, such as RAM or SSD, for a performance boost."
    ),
    "linux network issues": (
        "To troubleshoot Linux network issues:\n"
        "1. Check your network interface with 'ip a' or 'ifconfig'.\n"
        "2. Restart the network service with 'sudo systemctl restart NetworkManager'.\n"
        "3. Check if your firewall is blocking the connection (use 'ufw' or 'firewalld').\n"
        "4. Inspect DNS settings in '/etc/resolv.conf'.\n"
        "5. Update or reinstall network drivers if needed."
    ),
    "linux package update": (
        "To update Linux packages and software:\n"
        "1. For Debian-based distros: 'sudo apt update && sudo apt upgrade'.\n"
        "2. For Red Hat-based distros: 'sudo dnf update' or 'sudo yum update'.\n"
        "3. For Arch-based distros: 'sudo pacman -Syu'.\n"
        "4. Check if your repositories are up to date and working correctly.\n"
        "5. Reboot after major kernel or system updates."
    ),
    "linux disk issues": (
        "If you're having disk-related issues on Linux:\n"
        "1. Check disk usage with 'df -h' or 'du -sh' for specific folders.\n"
        "2. Use 'fsck' to check and repair file system errors.\n"
        "3. Mount disks manually if they aren't recognized (use 'mount' command).\n"
        "4. Check disk health using 'smartctl' from 'smartmontools' package."
    ),

    # General Issues (all OS)
    "battery draining": (
        "To improve battery life:\n"
        "1. Reduce screen brightness.\n"
        "2. Close unused applications and background processes.\n"
        "3. Turn off Bluetooth and Wi-Fi when not in use.\n"
        "4. Adjust power settings (use power saver modes).\n"
        "5. For laptops, calibrate the battery by fully charging and discharging it."
    ),
    "overheating": (
        "To troubleshoot overheating:\n"
        "1. Check for dust in your device's fans and vents.\n"
        "2. Use your device on a flat, hard surface to improve airflow.\n"
        "3. Monitor CPU and GPU temperatures with third-party tools.\n"
        "4. Lower the performance settings or enable battery saver mode.\n"
        "5. Apply fresh thermal paste if your device is old and continues to overheat."
    ),
}

# Function to generate responses using the question-answering model (DistilBERT)
def get_help(query, context_text):
    try:
        response = qa_model(question=query, context=context_text)
        if response['score'] < 0.3:  # Confidence threshold
            return ""
        return response['answer']
    except Exception as e:
        logger.error(f"Error generating response with DistilBERT: {e}")
        return ""

# Function to handle /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    user_profiles[user_id] = {"os_version": None, "help_type": None, "answer_type": None}  # Initialize user profile
    await update.message.reply_text(
        'Welcome to OS Help Bot! Before we start, could you please tell me what operating system you are using?'
    )

# Function to handle initial user responses
async def handle_initial_response(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text.lower()
    user_id = update.message.from_user.id
    user_profile = user_profiles[user_id]

    if user_profile['os_version'] is None:
        user_profiles[user_id]['os_version'] = user_message.strip()
        await update.message.reply_text("Thanks! Are you looking for help with technical troubleshooting or theoretical concepts?")
        return

    if user_profile['help_type'] is None:
        user_profiles[user_id]['help_type'] = "technical" if "technical" in user_message else "theoretical"
        await update.message.reply_text("Got it! Do you prefer short, concise answers, or more detailed explanations?")
        return

    if user_profile['answer_type'] is None:
        user_profiles[user_id]['answer_type'] = "brief" if "short" in user_message else "detailed"
        await update.message.reply_text("Great! Now feel free to ask your questions.")
        return

    await handle_message(update, context)

# Function to handle text messages (updated with "Who created you?" feature + Google search summarization + GPT-Neo refinement)
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text.lower()
    user_id = update.message.from_user.id

    # Detect if the user is asking who created the bot
    if "who created you" in user_message or "who is your creator" in user_message:
        await update.message.reply_text("MUKKA SRIVATSAV and team has created me.")
        return
    if "who are all in that team" in user_message or "who are in team" in user_message:
        await update.message.reply_text("MUKKA SRIVATSAV\n K.VENKATESH\n M.VAISHNAVI\n A.SAISREE\n")
        return

    # Retrieve the user's OS version and help type
    user_profile = user_profiles.get(user_id, {})
    os_version = user_profile.get('os_version', 'generic OS')

    # Detect changes in help type mid-conversation
    if "technical" in user_message and user_profile['help_type'] != "technical":
        user_profiles[user_id]['help_type'] = "technical"
        await update.message.reply_text("You've switched to technical help. How can I assist you with technical issues?")
        return
    elif "theoretical" in user_message and user_profile['help_type'] != "theoretical":
        user_profiles[user_id]['help_type'] = "theoretical"
        await update.message.reply_text("You've switched to theoretical help. How can I assist you with theoretical concepts?")
        return

    # Check if the message matches an FAQ
    for question, answer in faq_data.items():
        if question in user_message:
            # Refine the FAQ response with GPT-Neo
            refined_answer = refine_with_gpt_neo(answer)
            await update.message.reply_text(f"Since you're using {os_version}, here is some advice:\n{refined_answer}")
            return

    # No FAQ match → Continue to other methods
    responses = []

    # Try to answer using the question-answering model (DistilBERT)
    qa_response = get_help(user_message, faq_context)
    if qa_response:
        responses.append(f"DistilBERT Answer:\n{qa_response}")

    # Use Google search for additional information
    google_summary, google_links = search_google_summary(user_message)
    if google_summary:
        responses.append(f"Google Search Summary:\n{google_summary}")
    
    # Combine and refine the final response with GPT-Neo
    combined_response = aggregate_responses(responses)
    refined_response = refine_with_gpt_neo(combined_response)

    # Send the refined response and search links to the user
    await update.message.reply_text(f"Based on your OS ({os_version}), here’s what I found:\n{refined_response}\n\nFor more information, you can check these resources:\n{google_links}")

# Refine the response using GPT-Neo
def refine_with_gpt_neo(response):
    try:
        # Use GPT-Neo to rephrase and refine the response
        result = gpt_neo_model(f"Rephrase this to be more user-friendly:\n{response}", max_length=150, do_sample=True, temperature=0.7)
        refined_text = result[0]['generated_text']

        # Clean up the response and return it
        cleaned_response = refined_text.strip().replace('\n', ' ')
        return cleaned_response
    except Exception as e:
        logger.error(f"Error refining response with GPT-Neo: {e}")
        return response  # If GPT-Neo fails, return the original response

# Google search and summarize results
def search_google_summary(query):
    try:
        # Search for the query and get the top 3 results
        search_results = list(search(query, num_results=3))  # Get top 3 results
        if not search_results:
            return "", ""
        
        summaries = []
        for url in search_results:
            # Scrape content from the URL
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Get the main content of the page
                paragraphs = soup.find_all('p')
                text = ' '.join([para.get_text() for para in paragraphs])
                
                # Add to summaries list (you can implement a more sophisticated summarization logic)
                summaries.append(text[:300])  # Get first 300 characters as a simple summary

            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
        
        # Combine the summaries
        combined_summary = "\n".join(summaries)

        # Collect links for further reading
        links = "\n".join([f"- {result}" for result in search_results])

        return combined_summary, links
    except Exception as e:
        logger.error(f"Error during Google search summarization: {e}")
        return "", ""

# Aggregate responses from different sources
def aggregate_responses(responses):
    unique_responses = set(responses)
    if len(unique_responses) == 0:
        return "I'm sorry, I couldn't find any relevant information."
    elif len(unique_responses) == 1:
        return unique_responses.pop()  # Return the single response
    else:
        combined_response = "\n".join(unique_responses)
        return f"Here are some insights:\n{combined_response}"

# Function to run the bot
async def run_bot():
    # Using the provided bot token
    token = '7552975521:AAF0PNVvR30FNEk6j1PH8aL9vz0-cKaMJJ8'  # Replace with your actual bot token

    # Build the application
    application = ApplicationBuilder().token(token).build()

    # Add handler for /start command
    application.add_handler(CommandHandler("start", start))

    # Add handler for messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_initial_response))

    # Start polling to receive updates and keep the bot running
    await application.run_polling()

# Main function
def main():
    asyncio.run(run_bot())  # Run the bot with asyncio.run()

if __name__ == '__main__':
    main()







